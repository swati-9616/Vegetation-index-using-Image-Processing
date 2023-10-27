# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:24:03 2022

@author: swati
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors
from skimage.io import imread, imshow
import random
import glob
from Function_List import *

filelist = glob.glob('Images\\'+'*.tif')
b=len(filelist)
multiimage = np.zeros((7681,7531,b))
for i in np.arange(b):
    multiimage[:,:,i] = imread(filelist[i])
r,c,b = np.shape(multiimage)    
imshow(multiimage[:,:,[1,2,3]]/65535)    
small_image = multiimage[4000:5000,4000:5000,:]
r,c,b = np.shape(small_image)
imshow(small_image[:,:,1:4]/65535)
data = list(map(lambda x: small_image[:,:,x].flatten(),np.arange(8)))
df = np.concatenate(data,axis = 0).reshape(8,1000000).T
k = 4 #(no of centres we need to select)
K = random.sample(range(0,1000000), k) #()
C =df[K,:]

#%% plot it
I_Class = np.zeros((r,c)) 
for i in np.arange(k):
    r0 =  clusterPoints[i]//c
    c0 = clusterPoints[i]%c
    I_Class[r0,c0] = i+1
    

colors = ["c","y","r","g"]
cmap = matplotlib.colors.ListedColormap(colors)
plt.figure()
ax3 = plt.imshow((I_Class), cmap = cmap)
plt.colorbar(ticks = range(1,k+1),label = 'Cluster Label')
plt.clim(0.5,4.5)

t1 = time.time()
print(f'the run time is: ', (t1-t0), "sec")