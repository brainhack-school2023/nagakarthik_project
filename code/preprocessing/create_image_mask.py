import numpy as np
import argparse
import nibabel as nib

# parse command line arguments
parser = argparse.ArgumentParser(description='BIDSify the MS brain database.')
parser.add_argument('--image', type=str, default='image.nii.gz')
parser.add_argument('--threshold', type=float, default=0.0)

args = parser.parse_args()

input_path=args.image
output_path=f'{input_path.replace(".nii.gz","")}_mask.nii.gz'

input_img_nii = nib.load(input_path)
input_img = input_img_nii.get_fdata()
mask = np.zeros(input_img.shape)

mask[input_img > args.threshold] = 1

mask_nii = nib.Nifti1Image(mask, input_img_nii.affine, input_img_nii.header)
nib.save(mask_nii, output_path)
