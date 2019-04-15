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
