"Image data association using skicit learn"

"Import the library used in the project"

import skimage
from skimage import data
import os
from skimage import io
import pandas as pd
from natsort import natsorted, ns
from sklearn.preprocessing import MinMaxScaler
from skimage import util
from skimage import filters
from mpl_toolkits.mplot3d import Axes3D

from sklearn import metrics
from sklearn.cluster import MeanShift,estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks
from skimage import color
import skimage.io
import PIL
from PIL import Image
from numpy import asarray

"""The first step to any machine learning model is data preparation in the case
of image is image preprocessing which means changing images into matrices."""

"Importing images"
base_width = 360
image = Image.open('0000.jpg')
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
plt.imshow(image)