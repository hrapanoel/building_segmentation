import os
from os.path import join


def files_absolute_path(dir_path):
	"""Return a list of the absolute paths of files in a directory"""
	abs_path = []
	file_list = os.listdir(dir_path)
	for file in file_list:
		abs_path.append(join(dir_path, file))
	return abs_path

def get_file_name(file_path):
	"""Return filename without extension"""
	filename = os.path.basename(file_path)
	return os.path.splitext(filename)[0]

def get_outfile(file_path, output_path):
	'''Return the output file path given the input file path and output base path
	Args:
		file_path: path to file
		output_path: base of output file
	Returns:
		path to output mask
	'''

	file_name = os.path.basename(file_path)
	return join(output_path, file_name)

def save_mask(file_path, mask, meta):
	'''Save a mask as GeoTif at a given file path
	Args:
		file_path: path to save mask to
		mask: mask
		meta: metadata associated to mask
	'''

	with rasterio.open(file_path, "w", **meta) as dest:
		dest.write(mask.astype(rasterio.uint8, copy=False),1)
