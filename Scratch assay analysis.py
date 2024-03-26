# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:05:38 2022

@author: Hp
"""

import cv2 
from matplotlib import pyplot as plt 
from skimage import io
import numpy as np
from skimage.filters import threshold_multiotsu,threshold_otsu
from skimage.filters.rank import entropy
from skimage.morphology import disk


import glob


time=0
scale = 0.45   #micron/pixel

time_list=[]
area_list=[]

path = "list save images/*.*"

for file in glob.glob(path):
    img = io.imread(file , as_gray=True)
    entropy_img = entropy(img , disk(6))
    thresh = threshold_otsu(entropy_img) 
    binary  = entropy_img <= thresh
    
    scratch_area = np.sum(binary==1)
    scratch_area = scratch_area*((scale)**2)
    print("time",time,"hr ","scratch area",scratch_area)
    time_list.append(time)
    area_list.append(scratch_area)
    time=time+1
    
plt.plot(time_list,area_list,"bo")



