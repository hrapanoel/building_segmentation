import rasterio
from rasterio.mask import mask, raster_geometry_mask
import fiona
import rasterio.plot
from descartes import PolygonPatch
import geopandas as gpd
from shapely.geometry import Polygon, shape
from shapely import geometry
import numpy as np
import cv2
from os.path import join
import os
import pickle
import tensorflow as tf
from tensorflow import keras
import random
from keras import backend as K
import keras
from keras.optimizers import SGD
import pandas as pd
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef, jacard_coef_loss, jacard_coef
from utils import files_absolute_path, save_mask, get_outfile
from postprocessing import grow, denoise

keras.backend.set_image_data_format('channels_first')
keras.backend.clear_session()

INPUT_PATH = "/home/holy/internship/UPDLI/resized/"
OUTPUT_PREDICTED_MASK = "/home/holy/internship/UPDLI/output/predicted_masks/"

def make_prediction_cropped(model, X_train, initial_size=(224, 224), final_size=(192, 192), num_channels=3, num_masks=1):
	"""Pad the image, divide into tiles, make predictions and crop predictions. Then stitch all predictions
	together to form an image."""
	X_train = X_train/255
	shift = int((initial_size[0] - final_size[0]) / 2)
    
	height = X_train.shape[1]
	width = X_train.shape[2]

	if height % final_size[1] == 0:
		num_h_tiles = int(height / final_size[1])
	else:
		num_h_tiles = int(height / final_size[1]) + 1

	if width % final_size[1] == 0:
		num_w_tiles = int(width / final_size[1])
	else:
		num_w_tiles = int(width / final_size[1]) + 1

	rounded_height = num_h_tiles * final_size[0]
	rounded_width = num_w_tiles * final_size[0]

	padded_height = rounded_height + 2 * shift
	padded_width = rounded_width + 2 * shift

	padded = np.zeros((num_channels, padded_height, padded_width))

	padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas
	up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
	padded[:, :shift, shift:-shift] = up

 	lag = padded.shape[1] - height - shift
	bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
	padded[:, height + shift:, shift:-shift] = bottom

	left = padded[:, :, shift:2 * shift][:, :, ::-1]
	padded[:, :, :shift] = left

	lag = padded.shape[2] - width - shift
	right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]

	padded[:, :, width + shift:] = right

	h_start = range(0, padded_height, final_size[0])[:-1]
	assert len(h_start) == num_h_tiles

	w_start = range(0, padded_width, final_size[0])[:-1]
	assert len(w_start) == num_w_tiles

	temp = []
	for h in h_start:
		for w in w_start:
			temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]

	prediction = model.predict(np.array(temp))

	predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

	for j_h, h in enumerate(h_start):
		for j_w, w in enumerate(w_start):
			i = len(w_start) * j_h + j_w
			predicted_mask[:, h: h + final_size[0], w: w + final_size[0]] = prediction[i][:, shift:shift + final_size[0], shift:shift + final_size[1]]

	return predicted_mask[:, :height, :width]

def pred_mask(pr, threshold):
	'''Predicted mask according to threshold'''
	pr_cp = np.copy(pr)
	pr_cp[pr_cp < threshold]=0
	pr_cp[pr_cp >= threshold]=1
	return pr_cp

unet_model = ZF_UNET_224(weights="drive/My Drive/Colab Notebooks/UPDLI/model/batch_20_dim_224_epochs_200_steps_per_epoch_25_1_april_round3.best.hdf5")
unet_model.compile(optimizer=SGD(0.05, momentum=0.001, decay = 1e-6), loss=dice_coef_loss, metrics=[dice_coef])


file_list = files_absolute_path(INPUT_PATH)
for file in file_list:
	with rasterio.open(file, 'r') as ds:
		image = ds.read()  # read all raster values
		image_meta = ds.meta.copy()
		

	H = image.shape[1]
	W = image.shape[2]
	image_meta.update({"driver": "GTiff",
			"height": H,
			"width": W,
			"transform": ds.transform,
			"count":1})

	prediction = make_prediction_cropped(unet_model, image, initial_size=(224, 224),
											final_size=(224-20, 224-20),
											num_masks=1, num_channels=3)
	predicted_mask = pred_mask(prediction, 0.5)

	# Post-processing
	image_denoise = grow(predicted_mask, 10)
	image_grow = denoise(image_denoise, 30)

	# Save predicted mask
	out_file = get_outfile(file, OUTPUT_PREDICTED_MASK)
	save_mask(out_file, image_grow, image_meta)
