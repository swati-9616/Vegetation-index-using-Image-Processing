# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 12:37:22 2022

@author: swati
"""

#%% Call required Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
import glob
from Function_List import *
import collections
import time
import pprint
#     Band = int(i.split('\\')[-1].split('_')[-1][3])
#%%

# filelist = glob.glob('Multispectral data\\'+'*.TIF')
# RGB_arr = []
# NDVI_arr=[]
# 
    
# =============================================================================
#   # if np.logical_or(Band == 3,Band == 5):
# =============================================================================
       #if Band == 3:
           # R = imread(i)
        #else :
            #NIR = imread(i)
    #if 'R' in locals() and 'NIR' in locals():       
# =============================================================================
#         R = R.astype('float64')/(2**16-1)
#         NIR = NIR.astype('float64')/(2**16-1)
#         NDVI= (NIR-R)/(NIR+R)
#         img_arr.append(NDVI)   
#         del R , NIR
# =============================================================================
        
        
        
# fig, ax = plt.subplots()
# bound = [-1,-0.28,0.015,0.14,0.27,0.36,0.74]
# colors = ["k","c","w","y","g","m","r"]
# cmap = matplotlib.colors.ListedColormap(colors)
# norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
   
# sc=plt.imshow(NDVI, alpha=0.5 ,cmap=cmap, norm = norm)
# cbar = plt.colorbar(sc, spacing="proportional")
# plt.xticks(np.arange(0,n, 300))
# plt.yticks(np.arange(0,m,300))
# plt.xlabel('Column')
# plt.ylabel('Row')
# plt.title('NDVI')


#%% File reading and stacking 
t0 = time.time()  
filelist = glob.glob('Multispectral data\\'+'*.TIF')
RGB_arr = []
r,c = imread(filelist[0]).shape
img_arr = np.zeros((r,c,len(filelist)))
NDVI_arr=[]
for i in np.arange(len(filelist)):
    img_arr[:,:,i] = (imread(filelist[i])/256)

img_arr = img_arr.astype(dtype = 'uint8')    
#img = img_arr[:,:,1]
#imshow(img_arr[:,:,51:54])
#
#imshow(img)
# img = cv2.imread(filelist[0])
# print('All metadata:')
# pprint.pprint(filelist[0].getInfo())
for i in np.arange(int(len(filelist)/5)):
    RGB_arr.append(img_arr[:,:,i*5:i*5+5])
    
    
#%% Creating RGB images along with all indices    
# RGB_img

# for i in np.arange(len(RGB_arr)):
#     RGB_img = RGB_arr[i][:,:,0:3]/(2**16-1)    
#     R = RGB_arr[i][:,:,2].astype('float64')/(2**16-1)
#     IR = RGB_arr[i][:,:,4].astype('float64')/(2**16-1)
#     G = RGB_arr[i][:,:,1].astype('float64')/(2**16-1)
#     B = RGB_arr[i][:,:,0].astype('float64')/(2**16-1)
#     RE = RGB_arr[i][:,:,3].astype('float64')/(2**16-1)
for i in np.arange(len(RGB_arr)):
    RGB_img = RGB_arr[i][:,:,0:3]    
    R = RGB_arr[i][:,:,2].astype('float64')
    IR = RGB_arr[i][:,:,4].astype('float64')
    G = RGB_arr[i][:,:,1].astype('float64')
    B = RGB_arr[i][:,:,0].astype('float64')
    RE = RGB_arr[i][:,:,3].astype('float64')
    
    NDVI= (IR-R)/(IR+R)
    MSAVI2 = (2*IR+1-np.sqrt((2*IR+1)**2)-8*(IR-R))/2
    NDWI= (G-IR)/(G+IR)
    #EVI = (g*((IR-R)/(IR+C1*R-C2*B+L)))
    # Vegetation Indices
    NDRE = (IR-RE)/(IR+RE)
    GNDVI = (IR-G)/(IR+G)
    LCI = (IR-RE)/(IR+R)
    OSAVI = (IR-R)/(IR+R+0.16)
    # ndviname = 'ndvi'+str(i+1)+'.csv'
    # ndwiname = 'ndwi'+str(i+1)+'.csv'
    # msaviname = 'msavi'+str(i+1)+'.csv'
    # ndrename = 'ndre'+str(i+1)+'.csv'
    # gndviname = 'gndvi'+str(i+1)+'.csv'
    # lciname = 'lci'+str(i+1)+'.csv'
    # osaviname = 'osavi'+str(i+1)+'.csv'
    # # np.savetxt(ndviname, NDVI, delimiter = ',')
    # # np.savetxt(ndwiname, NDWI, delimiter = ',')
    # # np.savetxt(msaviname, MSAVI2, delimiter = ',')
    # np.savetxt(ndrename, NDRE, delimiter = ',')
    # np.savetxt(gndviname, GNDVI, delimiter = ',')
    # np.savetxt(lciname, LCI, delimiter = ',')
    # np.savetxt(osaviname, OSAVI, delimiter = ',')
    
    # RGB Image
    #fig1 = plt.subplot(2, 3, 1)
    plt.figure()
    imshow(RGB_img)
    # NDVI image
   # fig2 = plt.subplot(2, 3, 2)
    plt.figure()
    ax2 = plt.imshow((NDVI))
    bound = [-0.28,0.015,0.14,0.18,0.27,0.36,0.74]
    colors = ["c","b","y","g","#FF7F50","r"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
      
    sc=plt.imshow(NDVI, alpha=1 ,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    # plt.xticks(np.arange(0,n, 300))
    # plt.yticks(np.arange(0,m,300))    
    plt.title('NDVI')
    
  
    # NDWI image
    #fig3 = plt.subplot(2, 3, 3)
    plt.figure()
    ax3 = plt.imshow((NDWI))
    bound = [-1,-0.3,0.0,0.2,1.0]
    colors = ["y","r","g","b"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
       
    st = plt.imshow(NDWI, alpha=1 ,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    plt.title('NDWI')
    
    # MSAVI2 image
    
   # fig4 = plt.subplot(2, 3, 4)
    plt.figure()
    ax4 = plt.imshow((MSAVI2))
    bound = [-1,0.2,0.4,0.6]
    colors = ["r","y","g"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
       
    ms = plt.imshow(MSAVI2, alpha=1,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    plt.title('MSAVI2')
    
  # NDRE image

   # fig5 = plt.subplot(2, 3, 5)
    plt.figure()
    ax5 = plt.imshow((NDRE))
    bound = [-1,0.2,0.6,1]
    colors = ["y","g","r"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
       
    nd = plt.imshow(NDRE, alpha=1 ,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    plt.title('NDRE')    
    plt.figure()
    # ax6 = plt.imshow((EVI))
    # bound = [-1,0.1,0.4,1.0]
    # colors = ["y","g","r"]
    # cmap = matplotlib.colors.ListedColormap(colors)
    # norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
   
    # st = plt.imshow(NDWI, alpha=1 ,cmap=cmap, norm = norm)
    # cbar = plt.colorbar()
    # plt.title('EVI')   
  
    plt.figure()
    ax6 = plt.imshow((OSAVI))
    bound = [-1,-0.5,0.1,0.5,1]
    colors = ["b","y","g","r"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
    st = plt.imshow(OSAVI, alpha=1 ,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    plt.title('OSAVI')
    plt.figure()
    ax7 = plt.imshow((LCI))
    bound = [-1,-0.5,0.1,0.5,1.0]
    colors = ["y","b","r","g"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
       
    st = plt.imshow(LCI, alpha=1 ,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    plt.title('LCI')
    plt.figure()
    ax3 = plt.imshow((GNDVI))
    bound = [-1,-0.5,0.1,0.5,1.0]
    colors = ["#FF7F50","r","y","g"]
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(bound, len(colors))
       
    st = plt.imshow(GNDVI, alpha=1 ,cmap=cmap, norm = norm)
    cbar = plt.colorbar()
    plt.title('GNDVI')
t1 = time.time()
print(f'the run time is: ', (t1-t0), "sec")    
#%%

# #%% 
# Z = list(map(lambda i: ImageNo(filelist[i]),np.arange(1164)))

# frequency = collections.Counter(Z)
# r,c = np.where((NDVI >0.0) & (NDVI<0.35))
# # UH = NDVI[np.where((NDVI >0.2) & (NDVI<0.6))]
# r1 , c1 = np.where((NDVI >))

#%% unsupervised image classification 
#kmeans
def image_to_pandas(RGB_img):
    df = pd.DataFrame([RGB_img[:,:,0].flatten(),
                        RGB_img[:,:,1].flatten(),
                        RGB_img[:,:,2].flatten()]).T
    df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
    return df
df_RGB_img = image_to_pandas(RGB_img)
df_RGB_img.head(5)
plt.figure(num=None, figsize=(8, 6), dpi=600)
kmeans = KMeans(n_clusters=  4, random_state = 42).fit(df_RGB_img)
result = kmeans.labels_.reshape(RGB_img.shape[0],RGB_img.shape[1])
imshow(result, cmap='viridis')
# plt.show()
# D=np.sqrt(np.sum(df-C[0,:])**2,axis=1))
# #%%
# imshow((img_arr[:,:,45:48]/(2**16-1)))
# imshow((img_arr[:,:,[3,1,2]]/(2**16-1)))

# imshow(img_fcc[:,:,0].astype('uint8'))
