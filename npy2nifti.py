import numpy as np
from nipy import save_image, load_image 
from nipy.core.api import Image
import argparse

def convert_npy_to_nii(npy_file, base_nifti_filename):
  npy_data = np.load(npy_file).astype('uint8')
  bnifti = load_image(base_nifti_filename)
  img = Image.from_image(bnifti, data=npy_data)
  print (img.get_data().shape, img.get_data().max(), img.get_data().min(), img.get_data().dtype)
  save_image(img, npy_file[:-4] + '.nii')


parser = argparse.ArgumentParser(description='Convert .npy to .nii')
parser.add_argument('npy_file', metavar='npy_file', help='npy file for convert')
parser.add_argument('nii_file', metavar='nii_file', help='nii base file', default='')
args = parser.parse_args()

convert_npy_to_nii(args.npy_file, args.nii_file)
