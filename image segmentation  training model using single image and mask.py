# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:13:10 2022

@author: Hp
"""


"""
Outlines

Feature based segmentation using Random Forest
Demonstration using multiple training images
STEP 1: READ TRAINING IMAGES AND EXTRACT FEATURES 
STEP 2: READ LABELED IMAGES (MASKS) AND CREATE ANOTHER DATAFRAME
STEP 3: GET DATA READY FOR RANDOM FOREST (or other classifier)
STEP 4: DEFINE THE CLASSIFIER AND FIT THE MODEL USING TRAINING DATA
STEP 5: CHECK ACCURACY OF THE MODEL
STEP 6: SAVE MODEL FOR FUTURE USE
STEP 7: MAKE PREDICTION ON NEW IMAGES




"""






"""
Gabor and traditional filters for feature generation and 
Random Forest, SVM for classification. 
"""

"""
saron sy pehly 1 dataFrame bnain gyy empty data frame ta k hum images pr differnet filters laga k feature extract kr k apny data frame main store krain

es main hum yeh krain gy k pehly humm feature generate krain gy humm differnent filter lagain gy
1 image pr filter lagany sy pehly hamary pas jo image thi uski x,y ko multipy kr k dataframe k andr store kr dain gy
phr humm alag alag filters laga kr us k x,y ko multiply kr k calculate kr k apny data frame k ander store krty jain gyyy
jub hum ny sary feature different tareky sy extract kr liye to wo hamary pas sary features ready ho jain gy hamary data frame main



phr us k bad ab label deny hain apny dataframe ko jis sy hamara model classify/predict  kr k bta saky k label kya ha hamara
Label image hum ny rgb main sy blue channel alag kiye or us pr threhold apply kr k bnaya ha


phr jo label image thi us k x,y ko multiply kr k apny data frame k andr store Label column ka name rakh k us k ander values store kr deni ha x,y ko multiplr kr k
phr jub hamary features or label sub 1 data Frame main store hogaye too humm apny train data or test data alag alag krain gyy jis sy hum apny data ko fit kr k test kr sakain

model1
Random forrest classifier
hummm 1 model banain gy jis main humm apny data frame ko train kr sakain tah k hum new feature enter krain to hamara model classify kr saky ya bta saky k label kya ha

model2
Support Vector Machine
hummm 2sra model banain gy jis main humm apny data frame ko train kr sakain tah k hum new feature enter krain to hamara model classify kr saky ya bta saky k label kya ha

yeh 2 model hum es liye use krain gy ta k check kr sakain k konsa model acha kam krta ha random forrest ya support vector machine


phr jub hum ny koi model random forrest ya svm use kr k apna model train kr liya to usko jab hamara model train hojaye ga to 
pickle modeule ko use krty howy hum apny train kiye howy model ko save kr lain gy

apny trained model ko jab 1 bar tyar kr k  save kr liya phr hum usko kabhi B use kr sakty hain predict krny k liye apni orignal image sy...

"""



import cv2
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

img = cv2.imread("rgb.png",0)
#Here, if you have multichannel image then extract the right channel instead of converting the image to grey. 
#For example, if DAPI contains nuclei information, extract the DAPI channel image first. 

df = pd.DataFrame()   #yeh hum ny empty data frame bnaya jis main humm features store krain gyy

img2 = img.reshape(-1)

#Save original image pixels into a data frame. This is our Feature #1.
df["Orignal image"] = img2        #hum ny data frame main 1 orignal image k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k


kernels = []
#Generate Gabor features
num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
for theta in range(2):
    theta = theta/4.  * np.pi
    for sigma in (1,3):
        for lamda in np.arange(0 , np.pi , np.pi/4):
            for gamma in (0.05 , 0.5):
                
                phi = 0
                
                gabor_name = "Gabor"+str(num)
                ksize = 9
                kernel = cv2.getGaborKernel((ksize,ksize) , sigma , theta , lamda , gamma , phi , ktype = cv2.CV_32F )
                
                kernels.append(kernel)
                fimg = cv2.filter2D(img, cv2.CV_8UC3 , kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_name] = filtered_img
                num=num+1
                
                



#hum ny data frame main 1 canny edge  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
#Gerate OTHER FEATURES and add them to the data frame
canny_image = cv2.Canny(img, 100 , 255)      #Image, min and max values
canny_reshape = canny_image.reshape(-1)

df["canny edge"] = canny_reshape          #Add column to original dataframe





from skimage.filters import sobel , scharr , prewitt , roberts

#hum ny data frame main 1 sobel  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
sobel_img = sobel(img)
sobel_reshape = sobel_img.reshape(-1)
df["sobel"] = sobel_reshape


#hum ny data frame main 1 scharr  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
scharr_img = scharr(img)
scharr_reshape = scharr_img.reshape(-1)
df["scharr"] = scharr_reshape


#hum ny data frame main 1 prewitt  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
prewitt_img = prewitt(img)
prewitt_reshape = prewitt_img.reshape(-1)
df["prewitt"] = prewitt_reshape


#hum ny data frame main 1 roberts  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
roberts_img = roberts(img)
robert_reshape = roberts_img.reshape(-1)

df["roberts"] = robert_reshape


from scipy import ndimage as nd

#hum ny data frame main 1 gaussian filter  sigma=3  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
gaussian_img = nd.gaussian_filter(img, sigma=3)
gaussian_img_reshape = gaussian_img.reshape(-1)

df["gaussian sigma 3"] = gaussian_img_reshape


#hum ny data frame main 1 gaussian filter  sigma=7  k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k
gaussian_img2 = nd.gaussian_filter(img, sigma=7)
gaussian_img2_reshape = gaussian_img2.reshape(-1)

df["gaussian sigma 7"] = gaussian_img2_reshape



#hum ny data frame main 1 median filter k name sy 1 column banaya jis main sari values store kr dii x,y ko multiply kr k

median_img = nd.median_filter(img , size = 3)
median_img_reshape = median_img.reshape(-1)
df["median image"] = median_img_reshape

#VARIANCE with size=3
#variance_img = nd.generic_filter(img, np.var, size=3)
#variance_img1 = variance_img.reshape(-1)
#df['Variance s3'] = variance_img1  #Add column to original dataframe


######################################               


#Now, add a column in the data frame for the Labels
#For this, we need to import the labeled image

###LABEL IMAGE  jis k feature humm Data frame main Label k column main store krain gyy 
label_img = img1 = cv2.imread("threshold image.jpg",0)
label_img_reshape = label_img.reshape(-1)
df["label"] = label_img_reshape


orignal_image = df.drop("label" , axis = 1)   #Use for prediction              #label ka column add krany sy pehly hamari jo image thi wo orignal image thiii jisko humm train model k sath predict krain gyy es liye alag kiya ha hum ny

df = df[df.label != 0]


Y = df["label"].values

X = df.drop("label" , axis=1)




#Encode Y values to 0, 1, 2, 3, .... (NOt necessary but makes it easy to use other tools like ROC plots)
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)


#########################################################




from sklearn.model_selection import train_test_split

x_train , x_test , y_train, y_test = train_test_split(X , Y , test_size= 0.2  , random_state= 42)




# Import the model we are using
#RandomForestRegressor is for regression type of problems. 
#For classification we use RandomForestClassifier.
#Both yield similar results except for regressor the result is float
#and for classifier it is an integer. 

from sklearn.ensemble import RandomForestClassifier

# Instantiate model with n number of decision trees
model_rf = RandomForestClassifier(n_estimators=20 , random_state=41)

model_rf.fit(x_train , y_train)


feature_list = list(X.columns)
feature_imp = pd.Series(model_rf.feature_importances_ , index=feature_list).sort_values(ascending=False)




#Support vector machine
#SVM
# Train the Linear SVM to compare against Random Forest
#SVM will be slower than Random Forest. 
#Make sure to comment out Fetaure importances lines of code as it does not apply to SVM.

from sklearn .svm import LinearSVC
model_svm = LinearSVC(max_iter=100)
model_svm.fit(x_train, y_train)



#STEP 8: TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

#Test prediction on testing data. 


model_rf_predict = model_rf.predict(x_test)
model_svm_predict = model_svm.predict(x_test)



from sklearn import metrics

#Check accuracy on test dataset. If this is too low compared to train it indicates overfitting on training data.

print("Accuracy using random forest ", metrics.accuracy_score(y_test , model_rf_predict))
print("Accuracy using support vector machine" , metrics.accuracy_score(y_test,model_svm_predict))




#from yellowbrick.classifier import ROCAUC

#print("Classes in the images are ", np.unique(Y))

#unique = np.unique(Y)

#roc_auc = ROCAUC(model_rf  , classes = unique)
#roc_auc.fit(x_train , y_train)
#roc_auc.score(x_test , y_test)
#roc_auc.show()



#roc_auc = ROCAUC(model_svm  , classes = unique)
#roc_auc.fit(x_train , y_train)
#roc_auc.score(x_test , y_test)
#roc_auc.show()





#MAKE PREDICTION
#You can store the model for future use. In fact, this is how you do machine elarning
#Train on training images, validate on test images and deploy the model on unknown images. 



import pickle


#Save the trained model as pickle string to disk for future use
filename = "new feature extract"
pickle.dump(model_rf,open(filename ,  "wb"))


#To test the model on future datasets
loaded_img = pickle.load(open(filename , "rb"))
result = loaded_img.predict(orignal_image)

segmented = result.reshape((img.shape))


plt.imshow(segmented , cmap = "jet")


