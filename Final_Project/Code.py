#!/usr/bin/env python
# coding: utf-8
#Importing the Libraries
# In[1]:


import glob
import os
import sys
import warnings
from random import sample

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from hpelm import ELM
from mlxtend.plotting import plot_confusion_matrix
from PIL import Image, ImageChops
from scipy import ndimage
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             recall_score, roc_curve)
from sklearn.model_selection import StratifiedKFold


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


os.getcwd()




#Load the data

# In[4]:


print(os.listdir('TB dataset'))


# In[5]:


mont_tb_neg = glob.glob('TB dataset/images/TEST/*_0.jpg')
mont_tb_pos = glob.glob('TB dataset/images/TEST/*_1.jpg')


# In[6]:


mont_set = mont_tb_neg + mont_tb_pos
print(mont_set)


# In[9]:


mont_tb_neg_sample = sample(mont_tb_neg, 5)
mont_tb_pos_sample = sample(mont_tb_pos, 5)

#Display the data
# In[ ]:


mont_fig = plt.figure(figsize=(15, 8))
mont_fig.suptitle('Indian Tuberculosis DataSet', size=18)
for i, filename in enumerate(mont_tb_neg_sample + mont_tb_pos_sample):
    img = mpimg.imread(filename)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img, cmap='gray')
    if i <= 4:
        plt.title('Healthy')
    else:
        plt.title('Tuberculosis')
    

plt.show()

# Scale the Data
#Trimming
# In[ ]:


os.chmod('TB dataset/images/TEST/Trimmed/',0o777)  
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

from tqdm import tqdm
print(mont_set)
for filename in tqdm(mont_set):
    #print(filename)
    trim(Image.open(filename)).save(filename.replace('TEST', 'Trimmed'))

#Display the trimmed data
# In[ ]:


mont_set = glob.glob('TB dataset/images/Trimmed/*.png')


# In[ ]:


# Plot samples of trimmed  images
trim_fig = plt.figure(figsize=(15, 8))
trim_fig.suptitle('Indian Dataset Trimmed', size=18)
for i, filename in enumerate(mont_tb_neg_sample + mont_tb_pos_sample):
    plt.subplot(2, 5, i + 1)
    plt.imshow(
        np.array(Image.open(filename.replace('TEST', 'Trimmed'))),
        cmap='gray')
    if i <= 4:
        plt.title('Healthy')
    else:
        plt.title('Tuberculosis')

plt.show()


# In[ ]:

#Compress the data
os.chmod('TB dataset/images/Compressed/',0o777)

for i, filename in tqdm(enumerate(mont_set)):
    Image.open(filename).resize((1024, 1024), Image.ANTIALIAS).save(
        filename.replace('Trimmed', 'Compressed'))
0it [00:00, ?it/s]


# In[ ]:


# Plot samples of compressed images
mont_fig = plt.figure(figsize=(15, 8))
mont_fig.suptitle('Indian Dataset Compressed', size=18)
for i, filename in enumerate(mont_tb_neg_sample + mont_tb_pos_sample):
    plt.subplot(2, 5, i + 1)
    plt.imshow(
        np.array(Image.open(filename.replace('CXR_png', 'Compressed'))),
        cmap='gray')
    if i <= 4:
        plt.title('Healthy')
    else:
        plt.title('Tuberculosis')


# In[ ]:


import os
import PIL.ImageOps
 
for filename in glob.glob('TB dataset/images/Compressed/*.jpg'):
    PIL.ImageOps.invert(
        Image.open(filename).convert('L')).save('Data/Inverted/' +
                                                os.path.basename(filename))
    
    # Plot samples of inverted  images
mont_fig = plt.figure(figsize=(15, 8))
mont_fig.suptitle('Indian tb dataset Inverted', size=18)
for i, filename in enumerate(mont_tb_neg_sample + mont_tb_pos_sample):
    plt.subplot(2, 5, i + 1)
    plt.imshow(
        np.array(
            Image.open(filename.replace('Images/TEST', 'Inverted'))),
        cmap='gray')
    if i <= 4:
        plt.title('Healthy')
    else:
        plt.title('Tuberculosis')


# In[ ]:


import glob
import imageio

for image_path in glob.glob("TB dataset/images/TRAIN/Train_ne_X/*.jpg"):
    X_train = imageio.imread(image_path)
for image_path in glob.glob("TB dataset/images/TRAIN/Train_po_Y/*.jpg"):
    Y_train = imageio.imread(image_path)
for image_path in glob.glob("TB dataset/images/TEST/TEST_X/*.jpg"):
    X_test = imageio.imread(image_path)
for image_path in glob.glob("TB dataset/images/TEST/TEST_Y/*.jpg"):
    Y_test = imageio.imread(image_path)

#K-Means Classifer
# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error


# In[ ]:


Knn = KNeighborsClassifier()
Knn.fit(X_train, Y_train)
pred = Knn.predict(X_test)


# In[ ]:


train_predicts = Knn.predict(X_train)
test_predicts = Knn.predict(X_test)
print(np.sqrt(mean_squared_error(Y_train, train_predicts)))
print(np.sqrt(mean_squared_error(Y_test, test_predicts)))

#Decission Tree Regressor Model
# In[ ]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(min_samples_split=8, min_samples_leaf = 80)
fit = tree_reg.fit(X_train, Y_train)
pred = tree_reg.predict(X_test)

train_predicts = tree_reg.predict(X_train)
test_predicts = tree_reg.predict(X_test)
print(np.sqrt(mean_squared_error(Y_train, train_predicts)))
print(np.sqrt(mean_squared_error(Y_test, test_predicts)))







