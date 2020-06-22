# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 19:42:03 2020

@author: xphid
"""

# import libraries

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import imutils

cwd= os.getcwd()
print(cwd)

#%%

# Defining a variable for the path

image_path = "."

#%%

# put files into lists and return them as one list

def loadImages(path):
    image_files = sorted([os.path.join(path, 'cellule', file) for file in os.listdir(path
+ "/cellule")])
    return image_files

image_files = loadImages(image_path)

#%%
    
# Display two images

def display_2_img(a,b, title1 = "Original", title2 = "Edited"):
    plt.figure(figsize=(20,10))
    plt.subplot(121), plt.imshow((cv2.cvtColor(image, cv2.COLOR_BGR2RGB))), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b, cmap='gray', vmin=0, vmax=255), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()
   
#%%

# Loop over the images and plot the RGB and GRAY version

preprocessing=[]
lista_immagini=[]

for i in image_files:
    image = cv2.imread(i)
    lista_immagini.append(image)
    #cv2.imshow('Original', image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 
    #cv2.imshow('Gray Image', gray)
    #blurred = cv2.GaussianBlur(gray, (11, 11), 0)  #blurred
    display_2_img(image,gray,'Original','Gray Scale')
    preprocessing.append(gray[:])
                        
#%%

# save preprocessed images (RGB to GRAY) in a path named "Preprocessing"

if not os.path.exists('Preprocessing'):
    os.makedirs('Preprocessing')

for w,e in enumerate(preprocessing):
    title_prep= "./Preprocessing/Preprocessing" + str(w+1)+  ".jpg"
    prep_image= cv2.imwrite(title_prep , e)

#%%

# load the image and perform pyramid mean shift filtering to aid the thresholding step

im=list()
shifted_images=list()

for k in image_files:
    imag = cv2.imread(k)
    im.append(imag)
    shifted = cv2.pyrMeanShiftFiltering(imag, 21, 51)
    #cv2.imshow("Input", imag)
    shifted_images.append(shifted)
    
#%%

# convert the mean shift image to grayscale, then apply Otsu's thresholding
# save the thresh image in a path named "Thresh"

if not os.path.exists('Thresh'):
    os.makedirs('Thresh')

thresh_list=list()

for f in shifted_images:
    gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
    	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh_list.append(thresh)
    for ind_thr,thr in enumerate(thresh_list):
        #cv2.imshow("Thresh", thr)
        title_thresh= "./Thresh/Thresh" + str(ind_thr+1)+".jpg"
        cv2.imwrite(title_thresh, thr)

#%%

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map

total_DNA=list()

for ind,g in enumerate(thresh_list):
    D = ndimage.distance_transform_edt(g)
    localMax = peak_local_max(D, indices=False, min_distance=60,  # ho impostato min_distance a 60 perchè permette di riconoscere il DNA in modo pressappoco accurato in tutte le immagini 
    	labels=g)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm

    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=g)
    print("[image number_"+ str(1+ind)+"] {} DNA spots found".format(len(np.unique(labels)) - 1))
    total_DNA.append(len(np.unique(labels)) - 1)

#%%

# loop over the unique labels returned by the Watershed
# algorithm

for label in np.unique(labels):
    
    # if the label is zero, we are examining the 'background'
	# so simply ignore it
    if label == 0:
        continue
    
    # otherwise, allocate memory for the label region 
    # and draw it on the mask
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255

	# detect contours in the mask and grab the largest one
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
        		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    key=cv2.contourArea
    c = max(cnts, key=cv2.contourArea )
   
	# draw a circle enclosing the DNA spot
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.circle(imag, (int(x), int(y)), int(r), (0, 255, 0), 2)
   	
# show the output  (I draw a circle enclosing the DNA spot)
# NB: I draw circle enclosing the DNA spot just on the last image
cv2.imwrite("Image5_output.jpg", imag)

#%%

# Plot the original image and Otsu's thresholding

for d,f in zip(im , thresh_list):
        plt.figure(figsize=(20,10))
        images = [d, f]
        titles = ['Image',"Otsu's Thresholding"]
        plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
        plt.title(titles[0]), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(images[1] , cmap='gray')
        plt.title(titles[1]), plt.xticks([]), plt.yticks([])

#%%

# total area of DNA spot (in term of pixel)
# measure the average area according to DNA spots recognized

area_DNA=list()

for a in thresh_list:
    for b in total_DNA:
        average_area_DNA= (np.sum(np.array(a) != 0))//b
    area_DNA.append(average_area_DNA)
    
#%%

# create series 

index=[1,2,3,4,5]
serie1=pd.Series(image_files)
serie2= pd.Series(total_DNA)
serie3=pd.Series(area_DNA)

print(serie1)
print(serie2)
print(serie3)  

#%%

# create a dataset :
# 4 column: ( index, name of the image, number of DNA spots,
# average area of DNA spots in term of pixel )

df1 = pd.DataFrame({"Index": index , "Image": serie1 , "DNA spot": serie2, "Average_area": serie3})
    
print(df1)

#%%

# save the dataset in a csv file

df1.to_csv(r'.\cellule1.csv', index = False)

#%%

# create a new dataset :
# 3 column: ( index, name of the image, number of DNA spots)

df2 = pd.DataFrame({"Index": index , "Image": serie1 , "DNA spot": serie2})
    
print(df2)

#%%

# save the new dataset in a csv file

df2.to_csv(r'.\cellule2.csv', index = False)