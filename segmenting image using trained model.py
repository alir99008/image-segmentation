# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:45:19 2022

@author: Hp
"""

"""
pehly hum ny apna model trained kiya tha usko fatures or labels dy kr 

LAKIN

lakin es main hum ny trained model use kiya jo hum ny pehly train kiya tha phr new image jo hum ny predict krni us k features extract kiye
features extract krny k bad hum ny jo pehly model train kr k save kiya howa tha disk main us k sath es image k features ko predict krain gy


es main hum ny function use kiya ha  feature_extraction k name sy jis main image k sary features extract kr k 1 data frame main store kr k return kr dy ga

remeber ........ sirf features extract kry ga label nai Q k hum sirf trained model sy apni image k features ko predict kr rhy hain na k model train kr rhy hain

us k bad jo hum ny neechay path use kiya ha us k andr jitni image hongi sari images ko 1 , 1 kr k leta jay ga or tained model sy predict krta jay ga

"""

###############################################################
#STEP 7: MAKE PREDICTION ON NEW IMAGES
################################################################ 
import numpy as np
import cv2
import pandas as pd
 
def feature_extraction(img):
    df = pd.DataFrame()


#All features generated must match the way features are generated for TRAINING.
#Feature1 is our original image pixels
    img2 = img.reshape(-1)
    df['Original Image'] = img2

#Generate Gabor features
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(img2, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
########################################
#Geerate OTHER FEATURES and add them to the data frame
#Feature 3 is canny edge
    edges = cv2.Canny(img, 100,200)   #Image, min and max values
    edges1 = edges.reshape(-1)
    df['Canny Edge'] = edges1 #Add column to original dataframe

    from skimage.filters import roberts, sobel, scharr, prewitt

#Feature 4 is Roberts edge
    edge_roberts = roberts(img)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts'] = edge_roberts1

#Feature 5 is Sobel
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1

#Feature 6 is Scharr
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3

    #Feature 10 is Median with sigma=3
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #Feature 11 is Variance with size=3
#    variance_img = nd.generic_filter(img, np.var, size=3)
#    variance_img1 = variance_img.reshape(-1)
#    df['Variance s3'] = variance_img1  #Add column to original dataframe


    return df


#########################################################

#Applying trained model to segment multiple files. 

import pickle
#from matplotlib import pyplot as plt
from skimage import io

#filename = "sandstone_model"
filename = "new feature extract"
loaded_model = pickle.load(open(filename, 'rb'))

path = "imges to predict"
import os
for image in os.listdir(path):  #iterate through each file to perform some action
    print(image)
    img1= cv2.imread(path+image)
    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    
    #Call the feature extraction function.
    X = feature_extraction(img)
    result = loaded_model.predict(X)
    segmented = result.reshape((img.shape))
    segmented = segmented.astype(np.int8)
    io.imsave('imges to predict'+ image, segmented)
    #plt.imsave('images/sandstone/Segmanted_images/'+ image, segmented, cmap ='jet')
    
    
    
    