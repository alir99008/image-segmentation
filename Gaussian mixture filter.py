# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:40:05 2022

@author: Hp
"""



"""
NOTE:
Both BIC and AIC are included as built in methods as part of Scikit-Learn's  GaussianMixture. 
Therefore we do not need to import any other libraries to compute these. 
The way you compute them (for example BIC) is by fitting a GMM model and then calling the method BIC. 

"""     








#Gaussain is same like Kmean  but in gsussain we some have gaussian
#Gaussian is also unsupervised machine learning
#gaussain 1 he image ko jo attributes diye usi pr bar bar change kr k dekhay ga jab k kmean 1 he image dekhay ga


import numpy as np
import cv2
from matplotlib import pyplot as plt

#Use plant cells to demo the GMM on 2 components
#Use BSE_Image to demo it on 4 components
#USe alloy.jpg to demonstrate bic and how 2 is optimal for alloy

img = cv2.imread("BSE.tif")
#plt.imshow(img)
# Convert MxNx3 image into Kx3 where K=MxN
img2 = img.reshape((-1,3))  #-1 reshape means, in this case MxN       #(653,734,3)   hamari image main y x and 3 cannnels hoty hain yeh y*x   means 653*734 ko multipy kr k result (479302 , 3 ) img2 k variable k andr store kr dy ga

from sklearn.mixture import GaussianMixture as GMM

#covariance choices, full, tied, diag, spherical   "tied ki jaga hum yeh sub use kr sakty hain"
gmm_model = GMM(n_components=4, covariance_type='tied')  #tied works better than full
gmm_model.fit(img2)
gmm_labels = gmm_model.predict(img2)        #yeh predicted value hum es liye find krty hain Q k region main divide krny k bad hamain apni image ko orignal shape main B lana hota y,x,channel 

#Put numbers back to original shape so we can reconstruct segmented image

segmented = gmm_labels.reshape((img.shape[0], img.shape[1]))       #es sy hum apni predited values ko lety howy apni orignal shape main laty hain image ko
plt.imshow(segmented)





#BIC   n_components = regions hoty hain k hum ny apni image ko kitny regions main divide kiya ha lakin hamain nai pta hota k kitny regons main divide krna best hoga to yeh find krny k liye BIC use krty hain
#BIC  is used to find the best number of compoments to divide our image





#How to know the best number of components?
#Using Bayesian information criterion (BIC) to find the best number of components
import numpy as np
import cv2

img = cv2.imread("BSE.tif")
img2 = img.reshape((-1,3))

from sklearn.mixture import GaussianMixture as GMM

n = 4
gmm_model = GMM(n, covariance_type='tied').fit(img2)
#The above line generates GMM model for n=2
#Now let us call the bic method (or aic if you want).

bic_value = gmm_model.bic(img2)  #Remember to call the same model name from above)
print(bic_value)  #You should see bic for GMM model generated using n=2.
#Do this exercise for different n values and plot them to find the minimum.


#Now, to explain m.bic, here are the lines I used in the video. 
n_components = np.arange(1,10)
gmm_models = [GMM(n, covariance_type='tied').fit(img2) for n in n_components]
print("GMM model",gmm_models)
plt.plot(n_components, [m.bic(img2) for m in gmm_models], label='BIC')       #es graph sy hamaain pta chalta ha k kitny number of component best hain


