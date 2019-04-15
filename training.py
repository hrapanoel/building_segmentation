import keras
import numpy as np
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
import rasterio
from tensorboard import *
import pickle

keras.backend.set_image_data_format('channels_first')

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, tile_df, batch_size=32, dim=(112,112), n_channels=3, n_mask_channels=1,
					shuffle=True, augment=True):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.tile_df = tile_df
		self.n_channels = n_channels
		self.n_mask_channels = n_mask_channels
		self.shuffle = shuffle
		self.augment = augment
		self.on_epoch_end()
        
    def __len__(self):
		'Denotes the number of batches per epoch'
 		return int(np.floor(len(self.tile_df) / self.batch_size))
    
    def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		#print(indexes)
        
		# Find list of IDs
		batch_df = self.tile_df.iloc[indexes]
          
		# Generate data
		X, y = self.__form_batch(batch_df)

 		return X, y
        
	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.tile_df))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
        
	def __form_batch(self, filepaths_temp):
		X_batch = np.zeros((self.batch_size, self.n_channels, *self.dim))
		y_batch = np.zeros((self.batch_size, self.n_mask_channels, *self.dim,))
      
		for i in range(len(filepaths_temp)):
			row = filepaths_temp.iloc[i]
         
			#load image  
			X = get_crop(row, 'filepath')
			X = X/255.

			#load mask
			y = get_crop(row, 'mask_path')
          
			y_batch[i] = y
			X_batch[i] = X
         
          
			if self.augment:
				xb = X_batch[i]
				yb = y_batch[i]

			#if horizontal_flip:
				if np.random.random() < 0.5:
					xb = flip_axis(xb, 1)
					yb = flip_axis(yb, 1)

			#if vertical_flip:
				if np.random.random() < 0.5:
					xb = flip_axis(xb, 2)
					yb = flip_axis(yb, 2)

			#if swap_axis:
				if np.random.random() < 0.5:
					xb = xb.swapaxes(1, 2)
					yb = yb.swapaxes(1, 2)

				X_batch[i] = xb
				y_batch[i] = yb
		return X_batch, y_batch

def flip_axis(x, axis):
	x = np.asarray(x).swapaxes(axis, 0)
	x = x[::-1, ...]
	x = x.swapaxes(0, axis)
	return x
  
def get_crop(row, which):
	with rasterio.open(row[which]) as src:
 		#out_img, out_transform = mask(src, [row['geometry']], crop=True)
		l,b,r,u = rasterio.features.bounds(row['geometry'], north_up=True, transform= ~src.transform)
		out_img = src.read(window=((int(np.round(u)),int(np.round(b))),(int(np.round(l)),int(np.round(r)))))
	return out_img

# Generators parameters
params = {'dim': (224, 224),
          'batch_size': 20,
          'n_channels': 3,
          'n_mask_channels': 1,
         'augment': True,
          'shuffle': True}
params_validation  = {'dim': (224,224),
          'batch_size': 20,
          'n_channels': 3,
          'n_mask_channels': 1,
         'augment': False,
          'shuffle': True}

train_data = gpd.read_file("drive/My Drive/Colab Notebooks/UPDLI/data/tiles/resized_train_224.shp")
train, validation = train_test_split(train_data, test_size = 0.2, random_state = 27)

training_generator = DataGenerator(train, **params)
validation_generator = DataGenerator(validation, **params_validation)

keras.backend.clear_session()

weight_path="drive/My Drive/Colab Notebooks/UPDLI/model/batch_20_dim_224_epochs_200_steps_per_epoch_25_1_april_round3.best.hdf5".format('vgg_unet')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
							save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1,
									 mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", 
                      mode="min", 
                      patience=50) 
callbacks_list = [checkpoint, early, reduceLROnPlat, TensorBoardColabCallback(tbc)]

unet_model = ZF_UNET_224(weights=weight_path)
unet_model.compile(optimizer=SGD(0.05, momentum=0.001, decay = 1e-6), loss=dice_coef_loss, metrics=[dice_coef])
history = unet_model.fit_generator(generator=training_generator,
									steps_per_epoch=25,
									epochs=200,
									validation_data=validation_generator,
									callbacks=callbacks_list)

#save weights at end of training
filepath_final = weight_path.split(".")[0] + "_final.hdf5"
history.model.save_weights(filepath_final)

#save history
history_path = weight_path.split(".")[0] + "_history.pkl"
with open(history_path, 'wb') as f:
	pickle.dump(history.history, f)