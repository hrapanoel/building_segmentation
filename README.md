# Building segmentation from aerial imagery
Using [ZF_UNET_224](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model) to do building segmentation from aerial imagery.

# The data
- **Aerial images:** aerial images of the study area
- **Shapefile of building footrpints:** building footprints for training
- **Shapefile of training area:** since our aerial images contain both areas with and without building footprints, we need to know which area to use for training

# Data preprocessing
Before training the model, we need to preprocess the data in order to get an input suitable for the model.

1. **Create masks**

To generate the masks from the aerial images and building footprints shapefile, first set the input and output file directories in ```create_masks.py``` and run

```python create_masks.py -shp /path/to/building/footprints/shapefile```

2. **Create train and test area**

First set the necessary file directories in ```create_tiles.py``` and run

```python create_tiles.py --tile_type train_test```

This will create two shapefiles train and test containing tiles for training and testing

# Train and predict

3. **Train**

First set the training parameters (batch size, number of epochs, learning rate,...), output directory and pretrained weight if using one and run

```python training.py```

4. **Predict**

# Results postprocessing

5. **Post-processing**
