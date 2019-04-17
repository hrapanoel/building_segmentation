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
import pandas as pd

def denoise(mask, eps):
    """Removes noise from a mask.
    Args:
      mask: the mask to remove noise from.
      eps: the morphological operation's kernel size for noise removal, in pixel.
    Returns:
      The mask after applying denoising.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


def grow(mask, eps):
    """Grows a mask to fill in small holes, e.g. to establish connectivity.
    Args:
      mask: the mask to grow.
      eps: the morphological operation's kernel size for growing, in pixel.
    Returns:
      The mask after filling in small holes.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)

def contours(mask):
    """Extracts contours and the relationship between them from a binary mask.
    Args:
      mask: the binary mask to find contours in.
    Returns:
      The detected contours as a list of points and the contour hierarchy.
    Note: the hierarchy can be used to re-construct polygons with holes as one entity.
    """

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def simplify(polygon, eps):
    """Simplifies a polygon to minimize the polygon's vertices.
    Args:
      polygon: the polygon made up of a list of vertices.
      eps: the approximation accuracy as max. percentage of the arc length, in [0, 1]
    """

    assert 0 <= eps <= 1, "approximation accuracy is percentage in [0, 1]"

    epsilon = eps * cv2.arcLength(polygon, closed=True)
    return cv2.approxPolyDP(polygon, epsilon=epsilon, closed=True)

predicted_mask =
image_denoise = grow(predicted_mask, 10)
image_grow = denoise(image_denoise, 30)