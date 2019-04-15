import argparse
import rasterio
from rasterio.mask import mask, raster_geometry_mask
import fiona
import numpy as np
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import Polygon
from shapely import geometry
from os.path import join
import os
import pandas as pd
import sys
from utils import files_absolute_path

MASK_BASE_PATH = "/home/holy/internship/UPDLI/masks/"
INPUT_PATH = "/home/holy/internship/UPDLI/resized/"
OUTPUT_SHP_TILES_BASE = "/home/holy/internship/shapefiles/tiles"
TRAIN_AREA_SHAPEFILE = "/home/holy/internship/shapefiles/train_area/train_area.shp"
LAND_PARCELS_SHAPEFILE = "/home/holy/internship/shapefiles/land_parcels/land_parcels.shp"
crs_data = {'ellps': 'WGS84',
 'k': 1,
 'lat_0': 0,
 'lon_0': 19,
 'no_defs': True,
 'proj': 'tmerc',
 'units': 'm',
 'x_0': 0,
 'y_0': 0}

def get_crop_shape(row, which):
	with rasterio.open(row[which]) as src:
		out_img, out_transform = mask(src, [row['geometry']], crop=True)
	return out_img.shape

def splitImageIntoCells(filename, tileSize=224):
	'''Takes a Rasterio dataset and splits it into squares of dimensions tileSize * tileSize'''
	tiles = []
	with rasterio.open(filename) as src:
		width = src.width
		height = src.height
		for x in range(0, width, tileSize):
			for y in range(0, height, tileSize):
				x = min(x+tileSize, width) - tileSize
				y = min(y+tileSize, height) - tileSize
				geom = getTileGeom(src.transform, x, y, tileSize)
				tiles.append(geom)
	return tiles
                
def getTileGeom(transform, x, y, squareDim):
	'''Generate a bounding box from the pixel-wise coordinates using the original
	 datasets transform property'''
	corner1 = (x, y) * transform
	corner2 = (x + squareDim, y + squareDim) * transform
	return geometry.box(corner1[0], corner1[1],
		corner2[0], corner2[1])

def tile_dataframe(filepath, crs, tileSize=224):
	''''''
	print("tile_dataframe", tileSize)
	tiles_geom = splitImageIntoCells(filepath, tileSize)
	#create an empty GeoDataFrame
	tile_data = gpd.GeoDataFrame()
	# Create a new column called 'geometry' to the GeoDataFrame
	tile_data['filepath'] = [filepath] * len(tiles_geom)
	tile_data['geometry'] = tiles_geom
	tile_data.crs = crs
	return tile_data

def get_extent(filepath):
	'''Set the bounds of the crop box (bounds of the tiff image)'''
	with rasterio.open(filepath) as src:
		xmin, xmax, ymin, ymax = src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top 
		bounds = Polygon( [(xmin,ymin), (xmin, ymax), (xmax, ymax), (xmax,ymin)] )
		return bounds

def get_corresponding_mask(image_path):
	'''Get the mask file path corresponding to the image path'''
	file_name = os.path.basename(image_path)
	return join(MASK_BASE_PATH, file_name)

def create_train(crs, bounds, tile_data, train_area_shapefile ):
	'''Create training geo dataframe'''
	# Crop train area and take the part inside the bounding box
	train_area_crop_geom = gpd.read_file(train_area_shapefile)
	train_area_crop_geom.crs = crs
	train_area_crop_geom['geometry'] = train_area_crop_geom['geometry'].intersection(bounds)

	# train area inside the image bounds
	train_area = train_area_crop_geom[train_area_crop_geom.geometry.area>0]

	#take tiles inside train area
	train_tiles = gpd.sjoin(tile_data, train_area, op='within')
	train_tiles = train_tiles[['filepath', 'geometry']]
	train_tiles['mask_path'] = train_tiles['filepath'].map(get_corresponding_mask)
	return train_tiles

def create_test(crs, tile_data, train_tiles, land_parcels_shp):
	'''Create test geo dataframe'''
	land_parcels_geom = gpd.read_file(land_parcels_shp)
	land_parcels_geom.crs = crs
	# part of tile_data but not in train_tiles
	tiles_difference = gpd.overlay(tile_data, train_tiles, how='difference')
	test_tiles = gpd.sjoin(tiles_difference, land_parcels_geom, op='intersects', how='inner') #tiles are repeated here
	#drop duplicated tiles
	unique_test_tiles = test_tiles[~test_tiles.index.duplicated(keep='first')]
	return unique_test_tiles

def create_train_test(filepath, train_area_shapefile, land_parcels_shp, crs, tileSize):
	'''Create train and test geopanda dataframe for one image'''
	tiles = tile_dataframe(filepath, crs)
	bounds = get_extent(filepath)
	train = create_train(train_area_shapefile, crs, bounds, tiles)
	test = create_test(land_parcels_shp, crs, tiles, train)
	return train, test

def write_shp(df_list, crs, tileSize, output_base, set_type):
	'''Write a list of dataframes to a shapefile'''
	gpd_df = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=crs)
	filename = set_type + "_" + str(tileSize) + ".shp"
	outfile = join(output_base, "tiles_"+filename)
	gpd_df.to_file(outfile, driver ='ESRI Shapefile')


def create_shp_train_test(dir_path=INPUT_PATH, train_area_shapefile=TRAIN_AREA_SHAPEFILE, 
	land_parcels_shp=LAND_PARCELS_SHAPEFILE, crs=crs_data, tileSize=224,
	output_base=OUTPUT_SHP_TILES_BASE):
	'''Create train and test shapefile files containing path to image and tiles location (polygon)'''
	train_df_list = []
	test_df_list = []
	file_list = files_absolute_path(dir_path)
	for filepath in file_list:
		train_tiles, test_tiles = create_train_test(filepath, train_area_shapefile, land_parcels_shp, crs, tileSize)
		train_df_list.append(train_tiles)
		test_df_list.append(test_tiles)
	write_shp(train_df_list, crs, tileSize, output_base, "train")
	write_shp(test_df_list, crs, tileSize, output_base, "test")

def create_shp( dir_path=INPUT_PATH, crs=crs_data, tileSize=224, output_base=OUTPUT_SHP_TILES_BASE):
	'''Create all tiles for images'''
	df_list = []
	file_list = files_absolute_path(dir_path)
	for filepath in file_list:
		df_list.append(tile_dataframe(filepath, crs, tileSize=tileSize))
	write_shp(df_list, crs, tileSize, output_base, "all")

def main():
	parser = argparse.ArgumentParser(description="Create geopandas dataframe tiles")
	parser.add_argument("--tile_size", default=224,
        dest="tile_size",
		help="Tile sizes")
	parser.add_argument("--tile_type", dest="tile_type",
		help="Type of tiles: all to create all tiles, train_test to get the train and test tiles on images")
	args = parser.parse_args()
	if args.tile_type == 'train_test':
		create_shp_train_test(tileSize=args.tile_size)
	elif args.tile_type == 'all':
		create_shp(tileSize=args.tile_size)
	else:
		print("Unknow tile_type. Tile_type must be all or train_test")
		sys.exit(1)


if __name__ == "__main__":
	main()
# create dataframe
#filepath = "drive/My Drive/UPDLI/AP/2018_Feb_8cm_W46C_2.tif"