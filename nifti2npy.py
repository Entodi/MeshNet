import nipy
import numpy as np
import argparse

def convert_nii_2_npy(nii_file, npy_file=''):
	data = nipy.load_image(nii_file).get_data()
	if npy_file == '':
		npy_file = nii_file[:-4] + '.npy'
 		np.save(npy_file, data)
	else:
		np.save(npy_file, data)

parser = argparse.ArgumentParser(description='Convert .nii to .npy')
parser.add_argument('nii_file', metavar='nii_file', help='nii file for convert')
parser.add_argument('--npy_file', metavar='npy_file', help='npy output file', default='')
args = parser.parse_args()

convert_nii_2_npy(args.nii_file, args.npy_file)