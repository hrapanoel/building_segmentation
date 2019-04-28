"""Find coverage: area of predicted polygon inside land parcels"""

import rasterio
import geopandas as gpd

LAND_PARCELS_SHAPEFILE = "/home/holy/internship/shapefiles/land_parcels/land_parcels.shp"
PREDICTED_FOOTPRINTS = "/home/holy/internship/UPDLI/output/predicted_building_footprints.shp"

land_parcels_geom = gpd.read_file(LAND_PARCELS_SHAPEFILE)
predicted_poly = gpd.read_file(PREDICTED_FOOTPRINTS)
#take predicted inside land parcels
predicted_within = gpd.overlay(land_parcels_geom, predicted_poly, how='intersection')
predicted_within["Predicted_Building_Area"] = predicted_within["geometry"].area
pred_build_area = predicted_within.groupby(['OBJECTID','Shape_Area', 'Coverage'], as_index=False).agg({"Predicted_Building_Area": "sum"})
pred_build_area["Predicted_Coverage"] = (pred_build_area["Predicted_Building_Area"]/pred_build_area["Shape_Area"])*100
land_parcels_pred = pd.merge(land_parcels_geom, pred_build_area[['OBJECTID','Predicted_Building_Area','Predicted_Coverage']], how='left', on="OBJECTID", 
					left_on=None, right_on=None,left_index=False, right_index=False, sort=False,
					suffixes=('_x', '_y'), copy=True, indicator=False,
					validate=None)

land_parcels_pred.to_file(driver = 'ESRI Shapefile',
	filename= "/home/holy/internship/UPDLI/output/predicted_polygons/land_parcels_with_predictions.shp")