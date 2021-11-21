"Image data association using skicit learn"

"Import the library used in the project"

import skimage
from skimage import data
import os
from skimage import io
from natsort import natsorted, ns

from sklearn import metrics
from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt

"""The first step to any machine learning model is data preparation in the case
of image is image preprocessing which means changing images into matrices."""

# "Loading multiple images"
# list_files = os.listdir('entry-P10\images')
# list_files = natsorted(list_files)
# image_list = []
# for filename in list_files:
#   image_list.append(io.imread(filename))
"Importing images"
image=io.imread('0000.jpg')

"Converting image data into array"
data=np.asarray(image)

"printing or showing the image data"
plt.imshow(image)

"running the mean-shift model"
def mean_shift_fn(data=data, bandwidth=0.5):
    model = MeanShift(bandwidth=bandwidth).fit(data)
    cluster_centers= model.cluster_centers_
    return model 

ms=mean_shift_fn()