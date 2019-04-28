import rasterio
import geopandas as gpd
from utils import files_absolute_path

OUTPUT_PREDICTED_MASK = "/home/holy/internship/UPDLI/output/predicted_masks/"
PREDICTED_FOOTPRINTS = "/home/holy/internship/UPDLI/output/predicted_building_footprints"
crs_data = {'ellps': 'WGS84',
 'k': 1,
 'lat_0': 0,
 'lon_0': 19,
 'no_defs': True,
 'proj': 'tmerc',
 'units': 'm',
 'x_0': 0,
 'y_0': 0}

def predicted_polygon_shapefile(mask_dir=OUTPUT_PREDICTED_MASK, outfile = PREDICTED_FOOTPRINTS):
	predicted_polygons = []
	file_list = files_absolute_path(mask_dir)
	for file in file_list:
		predicted_polygons.append(mask_to_poly(file))
	write_shp(predicted_polygons, outfile)

def mask_to_poly(filename):
	with rasterio.open(filename, 'r') as src:
		#image = src.read()  # read all raster values
		shapes = rasterio.features.shapes(src.read()[0,...], transform=src.transform)
		# select the records from shapes where the value is 1,
		# or where the mask was True
		records = [{"geometry": geometry, "properties": {"value": value}}
					for (geometry, value) in shapes if value == 1]
		predicted_poly = gpd.GeoDataFrame.from_features(records)
		return predicted_poly

def write_shp(df_list, outfile, crs=crs_data):
	'''Write a list of dataframes to a shapefile
	Args:
		df_list: list of geopandas dataframe
		crs: coordinate system
	Returns:
		Write geopandas dataframe to shapefile
	'''
	gpd_df = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=crs)
	gpd_df.to_file(outfile, driver ='ESRI Shapefile')


