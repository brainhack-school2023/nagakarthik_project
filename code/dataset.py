from pathlib import Path
import os
import torch
from torch.utils.data import Dataset
from typing import Tuple
from dataset_utils import get_image_coordinate_grid_nib, norm_grid
import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class _BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(self, image_dir):
        super(_BaseDataset, self).__init__()
        self.image_dir = image_dir
        assert os.path.exists(image_dir), f"Image Directory does not exist: {image_dir}!"

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self) -> int:
        r"""Returns the number of coordinates stored in the dataset."""
        raise NotImplementedError

    def process(self):
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

class SyntheticSpineDataset(_BaseDataset):
    r""" Dataset of view1 and view2 T2w image sequence of the same patient.
    These could be e.g. an view1 and view2 T2w brain image,
    an view1 and view2 spine image, etc.
    However, both images must be registered to one another - the Dataset does not do this.
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
    """

    def __init__(self, image_dir: str="", name = "BrainLesionDataset",
                subject_id: str = "123456",
                session_id: str = "20230201", 
                view1_str: str='ax', 
                view2_str: str='sag',
                gt_str: str = 'iso',    # refers to the isotropic ground truth
                gt_mask_str: str = 'mask',    # refers to the mask of GT image
                transform = None, target_transform = None):
        super(SyntheticSpineDataset, self).__init__(image_dir)
        self.dataset_name = name
        self.subject_id = subject_id
        self.session_id = session_id
        self.view1_str = view1_str
        self.view2_str = view2_str
        self.gt_str = gt_str
        self.gt_mask_str = gt_mask_str

        self.dataset_name = (
            f'preprocessed_data/{self.dataset_name}_'
            f'{self.subject_id}_'
            f'{self.session_id}_'
            f'{self.view1_str}_'
            f'{self.view2_str}_'
            f'{self.gt_str}_'
            f'{self.gt_mask_str}'
            f'.pt'
        )

        print(self.dataset_name)

        files = sorted(list(Path(self.image_dir).rglob('*.nii.gz'))) 
        files = [str(x) for x in files]

        # only keep NIFTIs that follow specific subject 
        files = [k for k in files if self.subject_id in k]
        # print(files)

        view1 = [x for x in files if self.view1_str in x and self.subject_id in x and self.session_id in x][0]
        view2 = [x for x in files if self.view2_str in x and self.subject_id in x and self.session_id in x][0]
        gt = [x for x in files if self.gt_str in x and self.subject_id in x and self.session_id in x][0]

        self.view1 = view1
        self.view2 = view2
        self.gt = gt
        if self.gt_mask_str is not None:
            gt_mask = [x for x in files if self.gt_mask_str in x and 'mask' in x][0]
            self.gt_mask = gt_mask
        else:
            print("No GT mask provided. Continuing without computing SSIM/PSNR metrics")
            self.gt_mask = None

        if os.path.isfile(self.dataset_name):
            print("Dataset available.")
            dataset = torch.load(self.dataset_name)
            self.data = dataset["data"]
            self.label = dataset["label"]
            self.affine = dataset["affine"]
            self.dim = dataset["dim"]
            self.len = dataset["len"]
            self.coordinates = dataset["coordinates"]
            self.gt = dataset["gt"]
            if self.gt_mask is not None:
                self.gt_mask = dataset["gt_mask"]
            print("skipping preprocessing.")

        else:
            self.len = 0
            self.data = []
            self.label = []
            self._process()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx) -> Tuple[dict, dict]:
        data = self.data[idx]
        label = self.label[idx]
        return data, label

    def get_intensities(self):
        return self.label
    
    def get_coordinates(self):
        return self.coordinates

    def get_affine(self):
        return self.affine
    
    def get_dim(self):
        return self.dim
    
    def get_gt(self):
        # return nib.load(str(self.gt)).get_fdata()
        return self.gt
    
    def get_gt_mask(self):
        return self.gt_mask if self.gt_mask is not None else None
       
    def _process(self):

        print(f"Using {self.view1} as view1.")
        print(f"Using {self.view2} as view2.")
        if self.gt_mask is not None:
            print(f"Using {self.gt_mask} as gt mask.")

        # get the coordinates, label intensities and coordinate boundaries for each view
        view1_dict = get_image_coordinate_grid_nib(nib.load(str(self.view1)))
        view2_dict = get_image_coordinate_grid_nib(nib.load(str(self.view2)))

        data_view1 = view1_dict["coordinates"]
        data_view2 = view2_dict["coordinates"]

        min1, max1 = data_view1.min(), data_view1.max()
        print(f"Min and Max Coordinates of view1: {min1}, {max1}")
        min2, max2 = data_view2.min(), data_view2.max()
        print(f"Min and Max Coordinates of view2: {min2}, {max2}")

        min_c, max_c = np.min(np.array([min1, min2])), np.max(np.array([max1, max2]))
        print(f"Min and Max Coordinates out of both views: {min_c}, {max_c}")

        # re-normalize coordinates depending the lowest min and highest max between the two views
        data_view1 = norm_grid(data_view1, xmin=min_c, xmax=max_c)
        data_view2 = norm_grid(data_view2, xmin=min_c, xmax=max_c)

        labels_view1 = view1_dict["intensity_norm"]
        labels_view2 = view2_dict["intensity_norm"]
        
        # assemble the data and labels
        self.data = torch.cat((data_view1, data_view2), dim=0)
        self.label = torch.cat((labels_view1.flatten(), labels_view2.flatten()), dim=0)
        self.len = len(self.label)

        # store the GT images to compute SSIM and other metrics!
        gt_dict = get_image_coordinate_grid_nib(nib.load(str(self.gt)))
        self.gt = gt_dict["intensity_norm"]

        if self.gt_mask is not None:
            self.gt_mask = torch.tensor(nib.load(self.gt_mask).get_fdata()).bool()

        self.coordinates = gt_dict["coordinates_norm"]
        self.affine = gt_dict["affine"]
        self.dim = gt_dict["dim"]

        print(f"(coordinate) input data shape: {self.data.shape}")
        print(f"label intensities shape: {self.label.shape}")

        # store to avoid preprocessing
        dataset = {
            'len': self.len,
            'data': self.data,
            'label': self.label,
            'affine': self.affine,
            'gt': self.gt,
            'gt_mask': self.gt_mask if self.gt_mask is not None else None,
            'dim': self.dim,
            'coordinates': self.coordinates,
        }
        if not os.path.exists(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0])):
            os.makedirs(os.path.join(os.getcwd(), os.path.split(self.dataset_name)[0]))
        torch.save(dataset, self.dataset_name)

class InferDataset(Dataset):
    def __init__(self, grid):
        super(InferDataset, self,).__init__()
        self.grid = grid

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        data = self.grid[idx]
        return data

if __name__ == '__main__':

    dataset = SyntheticSpineDataset(
                image_dir='miccai',
                name='miccai_dataset',          
                )

    print("Passed.")
    
