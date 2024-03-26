# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:45:12 2022

@author: Hp
"""
#tecture hum es liye use krty hain jab threshold sy hum apni picture amin area nai select kr sakty tab hum texture use hain yeh bht minimum difference main B kam krta ha jahan pr threshold nai krta 

import cv2 
from matplotlib import pyplot as plt 
from skimage import io
import numpy as np
from skimage.filters import threshold_multiotsu,threshold_otsu


img = io.imread("texture.jpg" , as_gray=True)
#plt.imshow(img , cmap="gray")

from skimage.filters.rank import entropy
from skimage.morphology import disk


#Entropy es liye use krty hain hum Q k yeh pehly bht minmum difference ko find kr k us main different color fill kr deta ha jis sy threshold find krna easy ho jata ga
entropy_img = entropy(img , disk(6))     #yeh hum ny img py entropy function lagaya ha or kernal size 6 rakha ha jiska name disk ha jesa k  peechay B hum ny kiya tha
#plt.imshow(entropy_img)

#plt.hist(entropy_img.flat , bins=100 , range=(0,5))
thresh = threshold_otsu(entropy_img)   #phr hum ny entopy image jo find kii thi uska auto threshold nikala
#plt.show()

binary  = entropy_img <= thresh    #or jo values thresh sy kam thi usko binary main store krwa kr uski picture show krwa dii
plt.imshow(binary)
