# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 21:35:29 2022

@author: Hp
"""
#threshholding   : thresh hold hamary image ko binary main show krta ha image ka jo portion hum ny dekhna hota jonsy particles hum dekhna chahty hain usko binary main sjow krta ha


import cv2 
from matplotlib import pyplot as plt 
img = cv2.imread("rgb.png",1)
#plt.imshow(img)    
#plt.hist(img.flat , bins=100 , range=(0,255))           #hum ny jo hamari real image uska histogram show krwaya ha

blue_channel = img[: , : , 0]                               #es main hum ny apni image k blues collours ko select kiya ha
#plt.imshow(blue_channel , cmap = "gray")

#plt.hist(blue_channel.flat,bins = 100 , range = (0,255))        #phr select kiye gaye blue colors sy jo image ban rhi thi uska histogram show krwaya ha
#plt.show()


background = (blue_channel>=100)          #es main hum ny 100 sy uper jonsy colors blues main aaty hain jo histogram main show ho rhy thy usko background waly variable main store krwaya ha Q k wo color bht zada light thy es liye usko hum show nai krwana chahty thy         
nuclei = (blue_channel<100)              #es main hum ny 100 sy neechay jo histogram main show ho rhy thy 100 sy neechay jo blue colors main aaty hain dark colors unko nuclei waly variables main store krwa diya ha 
#plt.hist(nuclei.flat,bins = 100 , range = (0,150))        



#plt.imshow(nuclei , cmap="gray")

#es thresh hold main jo value 100 sy bari hogi wo white ho jay gi baki sari black yehi binary image hoti ha...
ret , thresh1 = cv2.threshold(blue_channel,100, 255, cv2.THRESH_BINARY)         # es main hamary pas 2 variables 1 wo variable jis main hamary pas wo variable store hoti jis main hamary pas thresh hold sy uper wali sari value hoti hain jesy ret = 150 ka matlab 150 sy uper wali sari values 255 main matlab white main convert kr dy or jo new image bany gi wo thresh1 main store hogi or cv2.Binary ka matlb ka wo picture binary main hogi 
plt.imshow(thresh1 , cmap="gray")


cv2.imwrite("threshold image.jpg",thresh1)
#auto thresh holding
ret2 , thresh2 = cv2.threshold(blue_channel,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    # es main 0 - 255 tk auto threshhold select kry ga or wo ret2 main store kr dy ga or thresh hold selsct hony k bad jo value thresh hold sy agy jo value hogi uski jo picture hogi wo thresh2 main store ho jay gi 
#plt.imshow(thresh2 , cmap="gray")
print(ret2)
plt.show()