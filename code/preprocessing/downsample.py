import argparse
import nibabel as nib
import nibabel.processing

# parse command line arguments
parser = argparse.ArgumentParser(description='BIDSify the MS brain database.')
parser.add_argument('--image', type=str, default='image.nii.gz')
parser.add_argument('--xdim',  type=float, default=1.0)
parser.add_argument('--ydim',  type=float, default=1.0)
parser.add_argument('--zdim',  type=float, default=1.0)

args = parser.parse_args()

input_path=args.image
output_path=f'{input_path.replace(".nii.gz","")}_{args.xdim}_{args.ydim}_{args.zdim}.nii.gz'
voxel_size=[args.xdim, args.ydim, args.zdim]

input_img = nib.load(input_path)
print("Original Resolution:", input_img.header.get_zooms())
print("Original Dimensions:", input_img.header.get_data_shape())
print("Original voxel orientation:", nib.aff2axcodes(input_img.affine))
print("Original Affine: \n", input_img.affine)

resampled_img = nibabel.processing.resample_to_output(in_img=input_img, voxel_sizes=voxel_size, order=3, mode='constant')
nib.save(resampled_img, output_path)

print("Resampled Resolution:", resampled_img.header.get_zooms())
print("Resampled Dimensions:", resampled_img.header.get_data_shape())
print("Resampled voxel orientation:", nib.aff2axcodes(resampled_img.affine))
print("Resampled Affine: \n", resampled_img.affine)

