import argparse
import rasterio
from rasterio.mask import mask, raster_geometry_mask
import fiona
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from utils import files_absolute_path
import os
from os.path import join

MASK_BASE_PATH = "/home/holy/internship/UPDLI/masks/"
INPUT_PATH = "/home/holy/internship/UPDLI/resized/"
BUILDING_SHAPEFILE = "/home/holy/internship/shapefiles/building_footprints/building_footprints_2d.shp"

def create_masks(bldg_footprints_shp=BUILDING_SHAPEFILE):
	'''Create masks for images in a directory given the building footprints shapefile
	Args:
		bldg_footprints_shp: building footprint shapefile
	'''
	files = files_absolute_path(INPUT_PATH)
	with fiona.open(bldg_footprints_shp, "r") as shapefile:
		geoms = [feature["geometry"] for feature in shapefile]
	for file in files:
		out_file = get_outfile(file)
		mask, mask_meta = create_one_mask(file, geoms)
		save_mask(out_file, mask, mask_meta)

def create_one_mask(file, geom):
	'''Create a mask for an image given the path and a list of geometries.
	Args:
		file: path to filename
		geom: geometry list from a shapefile
	Returns:
		Tuples of image mask and metadata associated to image
	'''
	with rasterio.open(file) as src:
		out_image, out_transform, window = raster_geometry_mask(src, geom, crop=False, invert=True)
		out_meta = src.meta.copy()
		out_meta.update({"driver": "GTiff",
			"height": out_image.shape[0],
			"width": out_image.shape[1],
			"transform": out_transform,
			"count":1})
	return out_image, out_meta

def get_outfile(file_path):
	'''Return the mask file path given the image file path
	Args:
		file_path: path to file
	Returns:
		path to output mask
	'''

	file_name = os.path.basename(file_path)
	return join(MASK_BASE_PATH, file_name)

def save_mask(file_path, mask, meta):
	'''Save a mask as GeoTif at a given file path
	Args:
		file_path: path to save mask to
		mask: mask
		meta: metadata associated to mask
	'''

	with rasterio.open(file_path, "w", **meta) as dest:
		dest.write(mask.astype(rasterio.uint8, copy=False),1)

def main():
	parser = argparse.ArgumentParser(description="Create masks from footprints shapefile")
	parser.add_argument("-shp", "--shapefile", default=BUILDING_SHAPEFILE,
        dest="shapefile",
		help="Footprints shapefiles.")
	args = parser.parse_args()
	create_masks(args.shapefile)

if __name__ == "__main__":
	main()