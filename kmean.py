# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 11:45:19 2022

@author: Hp
"""
#Gaussian is unsupervised machine learning
#kmean hum es liye use krty hain k hum apni image ko kis trahsegment krna chahty hain matlab hamary pas 1 image hain jis main 4 , 5 alag alag area hain lakin wo saii trah dekhai nai dy rhy to hum apni image ko saii trah alag alag colors main divide kr k achi trah segment krny k liye k mean use krty hain

#kmean different regions ko alag krta ha  or pattern find krny main B madad krta ha
#segmentation hamari udhr B kam aa sakti jahan pr hum sattelite sy li hoe image sy predict krna chahain k kahan kahan pani ha or kahan population ha


import cv2 
from matplotlib import pyplot as plt
from skimage import io


img = io.imread("BSE.tif")



img2 = img.reshape(-1,3) #-1 reshape means, in this case MxN     #(653,734,3)   hamari image main y x and 3 cannnels hoty hain yeh y*x   means 653*734 ko multipy kr k result (479302 , 3 ) img2 k variable k andr store kr dy ga
plt.imshow(img2)



from sklearn.cluster import KMeans

kmeans  =  KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)       #n_cluster = 4 ka matlab k hum apn image ko 4 regions main divide krna chahty hain bakii jitny B arguments hain sub by default hain

#note kmean hamsha reshape ki hoe image pr he appply ho sakta
model = kmeans.fit(img2)       #phr jo hum ny apna kemans ka function tyar kiya tha regions dy kr usko ko reshape ki hoe image main fit krna ha jis sy hamara model tyar hota ha


predicted_value = kmeans.predict(img2)     #yeh predicted value hum es liye find krty hain Q k region main divide krny k bad hamain apni image ko orignal shape main B lana hota y,x,channel 




seg_img = predicted_value.reshape((img.shape[0],img.shape[1]))        #es sy hum apni predited values ko lety howy apni orignal shape main laty hain image ko
plt.imshow(seg_img)



