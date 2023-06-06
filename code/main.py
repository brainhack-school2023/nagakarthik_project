import argparse
import time
import os
import yaml
import pathlib

import nibabel as nib
import numpy as np
import lpips

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

from model import MLPv1, MLPv2
from dataset import SyntheticSpineDataset, InferDataset
from sklearn.preprocessing import MinMaxScaler
from utils import input_mapping, compute_metrics, dict2obj, get_string, show_slices_gt



def parse_args():
    parser = argparse.ArgumentParser(description='Train Neural Implicit Function for a single scan.')
    parser.add_argument('--config', default='config.yaml', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument('--logging', action='store_true')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0], help="GPU ID following PCI order.")

    return parser.parse_args()


def main(args):

    # Init arguments 
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    # Load the config 
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config_dict)

    # logging run
    if args.logging:
        wandb.login()
        wandb.init(config=config_dict, project=config.SETTINGS.PROJECT_NAME, entity='naga-karthik',
                   name=config.SETTINGS.RUN_NAME, group=config.SETTINGS.GROUP_NAME)

    # make directory for models
    weight_dir = f'runs/{config.SETTINGS.PROJECT_NAME}/{config.SETTINGS.GROUP_NAME}_weights'
    image_dir = f'runs/{config.SETTINGS.PROJECT_NAME}/{config.SETTINGS.GROUP_NAME}_images'

    pathlib.Path(weight_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)

    # seeding
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)
    
    # device = f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu'
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')

    # load dataset
    dataset = SyntheticSpineDataset(
                    image_dir = config.SETTINGS.DIRECTORY,
                    name = config.SETTINGS.PROJECT_NAME,
                    subject_id=config.DATASET.SUBJECT_ID,
                    session_id=config.DATASET.SESSION_ID,
                    view1_str = config.DATASET.VIEW1_LABEL,
                    view2_str = config.DATASET.VIEW2_LABEL,
                    gt_str = config.DATASET.GT_LABEL,
                    )


    # Model Selection
    model_name = (
                f'{config.SETTINGS.PROJECT_NAME}_'
                f'{config.DATASET.SUBJECT_ID}_'
                f'{config.DATASET.SESSION_ID}_'
                f'view1-{config.DATASET.VIEW1_LABEL}_'
                f'view2-{config.DATASET.VIEW2_LABEL}_'
                f'gt-{config.DATASET.GT_LABEL}_'
                f'seed-{config.TRAINING.SEED}_'
    )

    # output_size, only one contrast!
    output_size = 1

    # Embeddings
    if config.MODEL.USE_FF:
        mapping_size = config.FOURIER.MAPPING_SIZE  # of FF
        input_size = 2 * mapping_size
        B_gauss = torch.tensor(np.random.normal(scale=config.FOURIER.FF_SCALE, size=(config.FOURIER.MAPPING_SIZE, 3)), dtype=torch.float32).to(device)
        input_mapper = input_mapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)
        model_name = f'{model_name}_FF_{get_string(config_dict["FOURIER"])}_'

    else:
        input_size = 3

    # Model Selection
    model = MLPv1(input_size=input_size, output_size=output_size, hidden_size=config.MODEL.HIDDEN_CHANNELS,
                num_layers=config.MODEL.NUM_LAYERS, dropout=config.MODEL.DROPOUT)
    model_name = f'{model_name}_MLP_'

    model.to(device)
    model_name = f'{model_name}_NUML_{config.MODEL.NUM_LAYERS}_HID_{config.MODEL.HIDDEN_CHANNELS}_D_{config.MODEL.DROPOUT}_'

    print(f'Number of MLP parameters {sum(p.numel() for p in model.parameters())}')

    # model for lpips metric
    lpips_loss = lpips.LPIPS(net='alex').to(device)
    
    # Loss
    if config.TRAINING.LOSS == 'L1Loss':
        criterion = nn.L1Loss()
    elif config.TRAINING.LOSS == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        raise ValueError('Loss function not defined!')

    model_name = f'{model_name}_{config.TRAINING.LOSS}' #__{config.TRAINING.LOSS_MSE_C1}__{config.TRAINING.LOSS_MSE_C2}_'

    # optimizer
    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)#, weight_decay=5e-5)
        model_name = f'{model_name}_{config.TRAINING.OPTIM}_{config.TRAINING.LR}_'    
    else:
        raise ValueError('Optim not defined!')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max= config.TRAINING.EPOCHS)
    
    mi_buffer = np.zeros((4,1))
    mi_mean = -1.0

    # Load Training Data
    train_dataloader = DataLoader(dataset, batch_size=config.TRAINING.BATCH_SIZE, 
                                 shuffle=config.TRAINING.SHUFFELING, 
                                 num_workers=config.SETTINGS.NUM_WORKERS)

    for epoch in range(config.TRAINING.EPOCHS):
        # set model to train
        model.train()
        wandb_epoch_dict = {}

        model_name_epoch = f'{model_name}_epch{int(epoch)}_model.pt'  
        model_path = os.path.join(weight_dir, model_name_epoch)

        print(model_path)

        loss_epoch = 0.0
        start = time.time()

        for batch_idx, (data, labels) in enumerate(train_dataloader):
            loss_batch = 0
            wandb_batch_dict = {}


            if torch.cuda.is_available():
                # RECALL: 
                # data is the normalized coordinates array, and 
                # labels in the normalized intensities array
                data, labels  = data.to(device=device), labels.to(device=device)

            if config.MODEL.USE_FF:
                data = input_mapper(data)
            elif config.MODEL.USE_SIREN:
                data = data*np.pi

            # pass to model
            target = model(data)

            if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL: # TODO check syntax compatibility
                target, _ = target

            loss = criterion(target.squeeze(), labels)
            
            if args.logging:
                wandb_batch_dict.update({'mse_loss': loss.item()})                    
                
            # zero gradients
            optimizer.zero_grad()
            # backprop
            loss.backward()
            optimizer.step()
            # epoch loss
            loss_batch += loss.item()
            loss_epoch += loss_batch

            if args.logging:
                wandb_batch_dict.update({'batch_loss': loss_batch})
                wandb.log(wandb_batch_dict)  # update logs per batch

        # collect epoch stats
        epoch_time = time.time() - start

        lr = optimizer. param_groups[0]["lr"]
        if args.logging:
            wandb_epoch_dict.update({'epoch_no': epoch})
            wandb_epoch_dict.update({'epoch_time': epoch_time})
            wandb_epoch_dict.update({'epoch_loss': loss_epoch})
            wandb_epoch_dict.update({'lr': lr})

        if epoch == (config.TRAINING.EPOCHS -1):
            torch.save(model.state_dict(), model_path)


        scheduler.step()
        ################ INFERENCE #######################

        model_inference = model
        model_inference.eval()

        # start inference
        start = time.time()

        # coordinate grid, affine and inference dataset are static
        # only process once!

        if epoch == 0:

            # assumes GT contrasts share common grid and affine
            mgrid = dataset.get_coordinates()
            print(f'Coordinates of GT data --> {mgrid.shape}')
            mgrid_affine = dataset.get_affine()     # can also use identity matrix
            x_dim, y_dim, z_dim = dataset.get_dim()
            print(f'Dimensions of GT data --> x_dim: {x_dim}, y_dim: {y_dim}, z_dim: {z_dim}')

            infer_data = InferDataset(mgrid)
            infer_loader = torch.utils.data.DataLoader(infer_data,
                                                       batch_size=4096,
                                                       shuffle=False,
                                                       num_workers=config.SETTINGS.NUM_WORKERS)

        out = np.zeros((int(x_dim*y_dim*z_dim), 2))
        model_inference.to(device)
        for batch_idx, (data) in enumerate(infer_loader):

            if torch.cuda.is_available():
                data = data.to(device)
                
            if config.MODEL.USE_FF:
                data = input_mapper(data)
            elif config.MODEL.USE_SIREN:
                data = data*np.pi
            else:
                data = data
                
            output = model_inference(data)

            if config.MODEL.USE_SIREN or config.MODEL.USE_WIRE_REAL:
                output, _ = output

            out[batch_idx*4096:(batch_idx*4096 + len(output)),:] = output.cpu().detach().numpy() 

        model_intensities=out

        ################ EVALUATION #######################

        model_intensities = model_intensities[:,0] 

        scaler = MinMaxScaler()
        label_arr = np.array(model_intensities, dtype=np.float32)
        model_intensities= scaler.fit_transform(label_arr.reshape(-1, 1))
        inference_time = time.time() - start
        if args.logging:
            wandb_epoch_dict.update({'inference_time': inference_time})

        print("Generating NIFTIs.")
        gt = dataset.get_gt().reshape((x_dim, y_dim, z_dim)).cpu().numpy()

        label_arr = np.array(gt, dtype=np.float32)
        gt = scaler.fit_transform(gt.reshape(-1, 1)).reshape((x_dim, y_dim, z_dim))

        img = model_intensities.reshape((x_dim, y_dim, z_dim))#.cpu().numpy()
        # pred = img
        # metrics = compute_metrics(gt=gt.copy(), pred=pred.copy(), mask=dataset.get_contrast1_gt_mask(), lpips_loss=lpips_loss, device=device)

        # if args.logging:
        #     wandb_epoch_dict.update({f'contrast2_ssim': metrics["ssim"]})
        #     wandb_epoch_dict.update({f'contrast2_psnr': metrics["psnr"]})
        #     wandb_epoch_dict.update({f'contrast2_lpips': metrics["lpips"]})

        nifti_name = model_name_epoch.replace("model.pt", f".nii.gz")

        # slices from the prediction
        slice_0 = img[int(x_dim/2), :, :]
        slice_1 = img[:, int(y_dim/2), :]
        slice_2 = img[:, :, int(z_dim/2)]

        # slices from the ground truth
        bslice_0 = gt[int(x_dim/2), :, :]
        bslice_1 = gt[:, int(y_dim/2), :]
        bslice_2 = gt[:, :, int(z_dim/2)]

        # plot slices
        im = show_slices_gt(
            pred_slices=[slice_0, slice_1, slice_2],
            gt_slices=[bslice_0, bslice_1, bslice_2], 
            epoch=epoch)
        
        if args.logging:
                image = wandb.Image(im, caption=f"prediction vs gt")
                wandb_epoch_dict.update({f"pred_image": image})

        affine = np.array(mgrid_affine)
        img = nib.Nifti1Image(img, affine)
        if epoch == (config.TRAINING.EPOCHS -1):
            nib.save(img, os.path.join(image_dir, nifti_name))
        
        if args.logging:
            wandb.log(wandb_epoch_dict)  # update logs per epoch


if __name__ == '__main__':
    args = parse_args()

    main(args)
