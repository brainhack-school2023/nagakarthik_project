import nibabel
import numpy as np
import torch
from tqdm import tqdm
# import SimpleITK as sitk
import torch.nn.functional as F
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler


# from Tancik et al.:
# https://github.com/tancik/fourier-feature-networks/blob/master/Experiments/3d_MRI.ipynb
# https://github.com/tancik/fourier-feature-networks/blob/master/Demo.ipynb
# Fourier feature mapping
def input_mapping(x, B):
    '''
    :param x: vector if input features
    :param B: matrix or None
    :return: 
    '''
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.T
        return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=-1)


def norm_grid(grid, xmin, xmax, smin=-1, smax=1):
    def min_max_scale(X, x_min, x_max, s_min, s_max):
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    return min_max_scale(X=grid, x_min=xmin, x_max=xmax, s_min=smin, s_max=smax)


def get_image_coordinate_grid_nib(image: nibabel.Nifti1Image, slice=False):
    """
    Given an NiFTI image, computes the coordinate grid and the corresponding label intensities
    Both the coordinates and the label arrays are then normalized between [-1, 1] and [0, 1] respectively
    Computes the boundary coordinates by applying the affine matrix to min and max coordinates in each dim. 

    """
    img_header = image.header
    img_data = image.get_fdata()
    img_affine = image.affine
    # Recall that the affine is 4x4 matrix that maps the voxel coordinates to the world coordinates. That is, 
    # going from voxel space (i,j,k) to world space (x,y,z) by applying the affine matrix.

    (x, y, z) = image.shape

    label = []
    coordinates = []

    for i in tqdm(range(x)):
        for j in range(y):
            for k in range(z):
                # simply applies affine matrix to points in voxel space to get world coordinates
                coordinates.append(nib.affines.apply_affine(img_affine, np.array(([i, j, k]))))
                # get the label intensity at that voxel
                label.append(img_data[i, j, k])

    # convert to numpy array
    coordinates_arr = np.array(coordinates, dtype=np.float32)
    label_arr = np.array(label, dtype=np.float32)   # NOTE: this is the array of label intensities not GT

    # coordinates_arr_norm = scaler.fit_transform(coordinates_arr)

    def min_max_scale(X, s_min, s_max):
        x_min, x_max = X.min(), X.max()
        return (X - x_min)/(x_max - x_min)*(s_max - s_min) + s_min

    coordinates_arr_norm = min_max_scale(X=coordinates_arr, s_min=-1, s_max=1)

    scaler = MinMaxScaler()

    label_arr_norm = scaler.fit_transform(label_arr.reshape(-1, 1))
            
    x_min, y_min, z_min = nib.affines.apply_affine(img_affine, np.array(([0, 0, 0])))
    x_max, y_max, z_max = nib.affines.apply_affine(img_affine, np.array(([x, y, z])))

    boundaries = dict()
    boundaries['xmin'] = x_min
    boundaries['ymin'] = y_min
    boundaries['zmin'] = z_min
    boundaries['xmax'] = x_max
    boundaries['ymax'] = y_max
    boundaries['zmax'] = z_max

    image_dict = {
        'boundaries': boundaries,
        'affine': torch.tensor(img_affine),
        'origin': torch.tensor(np.array([0])),
        'spacing': torch.tensor(np.array(img_header["pixdim"][1:4])),
        'dim': torch.tensor(np.array([x, y, z])),
        'intensity': torch.tensor(label_arr, dtype=torch.float32).view(-1, 1),
        'intensity_norm': torch.tensor(label_arr_norm, dtype=torch.float32).view(-1, 1),
        'coordinates': torch.tensor(coordinates_arr, dtype=torch.float32),
        'coordinates_norm': torch.tensor(coordinates_arr_norm, dtype=torch.float32),
    }
    return image_dict
