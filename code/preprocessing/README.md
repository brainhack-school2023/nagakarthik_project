This folder contains the files for preprocessing the data. The only preprocessing steps required to train an INR super-resolution model are: 

1. Create two different views of the same subject by downsampling it on different anatomical planes. 
2. Create a mask of the input image (which will later be used for computing the reconstruction accuracy)


The script `downsample.py` mainly uses `nibabel` to load and resample the data. The usage of the script is as follows:

```bash
python downsample.py --image <path_to_image> --xdim <x_dim> --ydim <y_dim> --zdim <z_dim>
```
where, `x_dim, y_dim, z_dim` are the voxel sizes defined in mm. For example, for downsampling the input image to 1mm x 1mm x 4mm resolution, the command would be:

```bash
python downsample.py --image <path_to_image> --xdim 1.0 --ydim 1.0 --zdim 4.0
```

The downsampled output will be saved in the same directory as the original image with the following naming convention: `<original_image_name>_<x_dim>_<y_dim>_<z_dim>.nii.gz`. These will then be used as inputs for training the model. 


The script `create_image_mask.py` creates a binary mask of the input image. The usage of the script is as follows:

```bash
python create_image_mask.py --image <path_to_image> --method min
```
where, `--method min` specifies that `skimage.filters.threshold_minimum` is used for automatic thresholding to create the mask. This is recommended over manually using a threshold value. 

