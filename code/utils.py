import torch
import numpy as np
import json
import torch.nn as nn
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import matplotlib.pyplot as plt 


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

'''
def get_string(my_dict):
    result = '_'.join([f"{key}-{value}" for key, value in my_dict.items()])
    return result
'''
def get_string(my_dict):
    return '_'.join([str(value) for value in my_dict.values()])


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def min_max_scale(X, s_min, s_max):
    x_min, x_max = X.min(), X.max()
    return torch.tensor((X - x_min) / (x_max - x_min) * (s_max - s_min) + s_min)


# from official FF repository/notebook
# https://colab.research.google.com/github/tancik/fourier-feature-networks/blob/master/Demo.ipynb
class input_mapping(nn.Module):
    def __init__(self, B=None, factor=1.0):
        super(input_mapping, self).__init__()
        self.B = factor * B
    
    def forward(self, x):

        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def compute_metrics(gt, pred, mask, lpips_loss, device):

    if type(mask) == torch.Tensor:
        mask = mask.float().cpu().numpy()

    assert mask.max() == 1.0, 'Mask Format incorrect.'
    assert mask.min() == 0.0, 'Mask Format incorrect.'

    gt -= gt[mask == 1].min()
    gt /= gt.max()
    gt *= mask

    pred -= pred[mask == 1].min()
    pred /= pred.max()
    pred *= mask

    ssim = structural_similarity(gt, pred, data_range=1)
    psnr = peak_signal_noise_ratio(gt, pred, data_range=1)

    x, y, z = pred.shape

    lpips_val = 0

    for i in range(x):
        pred_t = torch.tensor(pred[i,:,:]).reshape(1, y, z).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[i,:,:]).reshape(1, y, z).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    for i in range(y):
        pred_t = torch.tensor(pred[:,i,:]).reshape(1, x, z).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[:,i,:]).reshape(1, x, z).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    for i in range(z):
        pred_t = torch.tensor(pred[:,:,i]).reshape(1, x, y).repeat(3,1,1).to(device)
        gt_t = torch.tensor(gt[:,:,i]).reshape(1, x, y).repeat(3,1,1).to(device)
        lpips_val += lpips_loss(gt_t, pred_t)

    lpips_val /= (x+y+z)

    vals = {}
    vals["ssim"]= ssim
    vals["psnr"]= psnr
    vals["lpips"] = lpips_val.item()

    return vals


def show_slices_gt(pred_slices, gt_slices, epoch):
    """ Function to display row of image slices """
    plt.close()
    fig, axes = plt.subplots(2, len(pred_slices), dpi=175)

    for i, slice in enumerate(pred_slices):
        axes[0][i].imshow(slice.T, cmap="gray", origin="lower")
        axes[0][i].axis("off")
    for i, slice in enumerate(gt_slices):
        axes[1][i].imshow(slice.T, cmap="gray", origin="lower")
        axes[1][i].axis("off")
    
    plt.suptitle(f"Mid-sagittal Slices of Implicit Spine-Generic Spinal Cord after {epoch}.")
    # axes[0][0].set_title("Implicit Reconstruction")
    # axes[1][0].set_title("Ground Truth")
    plt.tight_layout()
    return fig
