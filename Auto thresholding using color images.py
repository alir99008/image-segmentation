# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 12:24:24 2022

@author: Hp
"""


import cv2 
from matplotlib import pyplot as plt 
import numpy as np
from skimage.filters import threshold_multiotsu
img = cv2.imread("rgb.png",0)

#plt.imshow(img)    
#plt.hist(img.flat , bins=100 , range=(0,255)) 
#plt.show()  



#yeh area hum es liye select kr rhy hain ta k es main hum colors fill kr sakain

region1 = (img >=0 ) & (img <= 75)     #0-75 area hum ny apni marzi sy area select kiya graph ko dekhty howy phr us k bad region 1 main store kr diya 
region2 = (img >75) & (img <= 140)    #76-140
region3 = (img >140)&(img <=200)   #141-200
region4 = (img > 200)&(img <=255)    #201-255


region = np.zeros((img.shape[0] , img.shape[1] , 3))   #matrix ki sari values zero kr dain gy       #(669, 953 , 3) humm ny pehly areas wise values select kr k pictures sy variabls main store kr dii Q k hum ny us main colors fill krny thy phr us k es liye pehly hummy img ki sari values ko zero krna pry ga es liye hum ny shape ko zero kiya(0 , 0 , 0) yeh es trah zero ho jain gy x , y ,3 channels sub zero ho jay ga



region[region1] = (1,0,0)    #es main ny region1 main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (1,0,0) ka matlab red color fill kr dain r, g, b
region[region2] = (0,1,0)     #es main ny region2 main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (0,1,0) ka matlab green color fill kr dain r, g, b
region[region3] = (0,0,1)      #es main ny region3 main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (0,0,1) ka matlab blue color fill kr dain r, g, b
region[region4] = (1,1,0)      #es main ny region1 main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (1,1,0) ka matlab yellow color fill kr dain r, g, b

#plt.hist(region.flat , bins=100 , range=(0,255))

#plt.imshow(region)    #es main hum color change hoe picture ko show krain gy k jin different region main hum ny clean kr k area wise selection kr k colors fill kiye thy
plt.show()




#Auto Thresholding
# auto threholding ka matlab yeh ha k yeh area auto select kry ga phr agr hum khud area select krain gy to hum main masla ho sakta ha matlab k hum ny area sahi trah divide na kiya hoo lakin auto thresh holding main yeh nai hota
#auto thresholding k liye hum library threshold_multiotsu use krty hain


thresholds = threshold_multiotsu(img , classes = 4)               # classes=4 ka matlab k hum ny 1 img ko 4 hiso main divide kiya or threshold waly variable main store kr diya
# IMP    " wesy to 1 line 51 main jo procedure how ha us sy hamari auto threshold complte to ho jati ha lain us k color achy nai hoty agr hum apni marzi sy colors fill krna chahty hain to phr neechay wala sara procedure kry hain means area select krty hain phr image ko clean krty hain phr select kiye area main colors fill krty hain wagera wagera......................


regionss = np.digitize(img , bins= thresholds)     #wo jo image ko 4 hison main divide kiya tha usko digitize wala np ka function use kr usko  0,1,2,3 ki form main regionss main store kr diya
#plt.imshow(regionss)


seg1 = (regionss == 0)         #jahan regionss == 0  hoga usko us area ko seg1 main store kr dy ga yeh wo area ha jo 4 classes banai thi matlab 4 hison main divide kiya tha
seg2 = (regionss == 1)         #jahan regionss == 1  hoga usko us area ko seg1 main store kr dy ga yeh wo area ha jo 4 classes banai thi matlab 4 hison main divide kiya tha
seg3 = (regionss == 2)          #jahan regionss == 2  hoga usko us area ko seg1 main store kr dy ga yeh wo area ha jo 4 classes banai thi matlab 4 hison main divide kiya tha
seg4 = (regionss == 3)          #jahan regionss == 3  hoga usko us area ko seg1 main store kr dy ga yeh wo area ha jo 4 classes banai thi matlab 4 hison main divide kiya tha

from scipy import ndimage as nd

seg1_opened = nd.binary_opening(seg1 , np.ones((3,3)))        #yeh binary opening ka function use kr k seg1 k area main 3*3 k matrix bana kr un saron ko 1 , 1 kr dy ga 
seg1_closed = nd.binary_closing(seg1_opened , np.ones((3,3)))     #jo binary open kr k seg1_opened main store kiya tha usko close krna lazmi hota ha es liye es main hum usko closed krain gy

seg2_opened = nd.binary_opening(seg2 , np.ones((3,3)))            #yeh binary opening ka function use kr k seg1 k area main 3*# k matrix bana kr un saron ko 1 , 1 kr dy ga 
seg2_closed = nd.binary_closing(seg2_opened , np.ones((3,3)))       #jo binary open kr k seg1_opened main store kiya tha usko close krna lazmi hota ha es liye es main hum usko closed krain gy

seg3_opened = nd.binary_opening(seg3 , np.ones((3,3)))
seg3_closed = nd.binary_closing(seg3_opened , np.ones((3,3)))

seg4_opened = nd.binary_opening(seg4 , np.ones((3,3)))
seg4_closed = nd.binary_closing(seg4_opened , np.ones((3,3)))


all_segment_cleaned = np.zeros((img.shape[0] , img.shape[1] , 3))      #matrix ki sari values zero kr dain gy       #(669, 953 , 3) humm ny pehly areas wise values select kr k pictures sy variabls main store kr dii Q k hum ny us main colors fill krny thy phr us k es liye pehly hummy img ki sari values ko zero krna pry ga es liye hum ny shape ko zero kiya(0 , 0 , 0) yeh es trah zero ho jain gy x , y ,3 channels sub zero ho jay ga

all_segment_cleaned[seg1_closed] = (1,0,0)        #es main ny seg1_closed main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (1,0,0) ka matlab red color fill kr dain r, g, b
all_segment_cleaned[seg2_closed] = (0,1,0)        #es main ny seg2_closed main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (1,0,0) ka matlab red color fill kr dain r, g, b
all_segment_cleaned[seg3_closed] = (0,0,1)        #es main ny seg3_closed main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (1,0,0) ka matlab red color fill kr dain r, g, b
all_segment_cleaned[seg4_closed] = (1,1,0)        #es main ny seg4_closed main jo area select kiya howa tha us area main  jo region picture jo clean ki thi us main (1,0,0) ka matlab red color fill kr dain r, g, b


#plt.imshow(all_segment_cleaned)
#plt.imsave("new auto threshold image save.jpg",all_segment_cleaned)