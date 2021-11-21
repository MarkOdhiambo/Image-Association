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

"Importing images and resizing images"

image = Image.open('0000.jpg')
base_width = 360
width_percent = (base_width / float(image.size[0]))
hsize = int((float(image.size[1]) * float(width_percent)))
image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
image = asarray(image)

"""Create a descriptor of the rgb of the image into rgb values corresponding to the rows and 
columns"""
index = pd.MultiIndex.from_product((*map(range, image.shape[:2]), ('r', 'g', 'b')),
    names=('row', 'col', None))
df_1 = pd.Series(image.flatten(), index=index)
df_1 = df_1.unstack()
df_1 = df_1.reset_index().reindex(columns=['col','row',   'r','g','b'])
df_1.head(10)

"""Vector 2 with only rgb values."""
df_2 = df_1[['r', 'g', 'b']]
df_2.head(10)

"Normalize the pixel  between value 0-1"
nd_1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(df_1)

"Calculating bandwidth"
bandwidth_1 = estimate_bandwidth(nd_1, quantile=.04)

""""After preprocessing run the data with and calculating the bandwith run the image
through the mean-shift algorithm."""
ms_1 = MeanShift(bandwidth = bandwidth_1 , bin_seeding=True, cluster_all=True,max_iter=100).fit(nd_1)

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

model_image=recreate_image(ms_1.cluster_centers_[:, 2:], ms_1.labels_, 240, 360)
"printing or showing the image data"
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original Image', loc='center')
plt.imshow(image)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Pixels with their location image ({} colors, Mean-Shift)'.format(len(ms_1.cluster_centers_)), loc='center')
plt.imshow(model_image);

