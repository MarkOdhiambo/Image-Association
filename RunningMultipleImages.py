# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:04:13 2021

@author: User
"""

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
from skimage.feature import hog
from skimage import data, color, io, exposure

"Importing images and resizing images"
data=[]

image = Image.open('0000.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0001.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0002.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0003.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0004.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0005.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0006.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

image = Image.open('0007.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)
fd, hog_image = hog(image, pixels_per_cell=(16, 16), block_norm='L2-Hys',visualize=True, multichannel=True)
data.append(fd)

"Normalize the data set"
data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)

"Calculating bandwidth"
bandwidth_1 = estimate_bandwidth(data, quantile=.04)

""""After preprocessing run the data with and calculating the bandwith run the image
through the mean-shift algorithm."""
ms_1 = MeanShift(bandwidth = bandwidth_1 , bin_seeding=True, cluster_all=True,max_iter=100).fit(data)
"""Recreating the image"""
def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

"Printing the number of clusters"
print("This is the number of clusters",len(ms_1.cluster_centers_))

# plt.figure(figsize=(12,10))

# plt.title('DAISY descriptors', fontsize=18)
# plt.imshow(descs_image)