# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:50:58 2022

@author: Hp
"""

#measure hum es liye use krty hain yeh pehly hamari picture main jo objects hain unko detect kr k har objecct main alag alag colors fill kr dy ga

from skimage import measure , io , img_as_ubyte
from matplotlib import pyplot as plt
from skimage.color import label2rgb , rgb2gray
import numpy as np
import cv2
from skimage.filters import threshold_otsu


img = cv2.imread("threshold image.jpg" , 0)
scale = 0.6

#plt.hist(img.flat , bins=100 , range=(0,255))
plt.show()


threshold = threshold_otsu(img)       #hum ny apni image pr auto threshold apply kiya ha

threshold_img = img<threshold       #hum ny apni image ny jahan threshold detect kiya tha or jo hamari orignal image main jo portion less then threhold hold ha usko select krain gy

#plt.imshow(threshold_img)


from skimage.segmentation import clear_border
#border_clear_image = clear_border(threshold)
#plt.imshow(border_clear_image , cmap="gray")

label_image = measure.label(threshold_img , connectivity=img.ndim)              #ab jo portio alag krny thy wo alag krny k bad jo hamari picture reh gai us pr measure.lebel ka filter appky kiya jis sy  hamari pictures main 2,3 trah k colurs fill ho jain gyy
#plt.imshow(label_image)



image_label_overlay = label2rgb(label_image , image = img)    #ab hum label2rgb sy jo picture hum ny label ki thi us main jo objects main us k colors k differnt tgb colors main convert krain gy
plt.imshow(image_label_overlay)               #yahan pr hum jis pictures main different colors show krway thy usko show krain gyy










#2ND METHOD

props = measure.regionprops_table(label_image , img , properties=['label' , 'area' , 'equivalent_diameter','mean_intensity','solidity'])


import pandas as pd
df = pd.DataFrame(props)
print(df.head())



df = df[df["area"]>50]
print(df.head())    #yeh hamary data ki first five show krta ha


#making new column in our dataset

df["area_square_microns" ] = df["area"]*(scale**2)
print(df.head())


