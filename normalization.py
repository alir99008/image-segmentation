# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 09:07:56 2022

@author: Hp
"""
# Nomalization hum es liye use krty hain Scaler hamari bht bari or choti values ko 0 sy 1 k beach main change kr deta ha jis sy predicton asan ho jati ha
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
 
df = pd.read_csv("breast cancer dataset.csv")

#print(df.head())        #  yeh hamary data set k har column ki first five lines print kry ga

#print(df["diagnosis"])        #hamary dataset main q diagosis wala column tha sirf us print kry ga

#Null values hum es liye check krty hain Q k agr hamary data set main 1 B null value ho to usko hamra dataset deal nai krta
#print(df.isnull().sum())      # yeh hamary har column main jitni null values honagi unko sum kr k tay ga

df = df.drop("Unnamed: 32" , axis=1)        #hamary data se main Unnamed: 32 ka column tha jis main null values thii es liye hum esko drop kr dain gyy
print(df)


#print(df.describe().T)

df = df.rename(columns = {"diagnosis":"Label"})     #hum apny diagnosis waly column ko as a Label rakh k deal krain gy gy es liye hm eska name change kr k label rakh dain gyy
print(df.head())


print(df["Label"].value_counts())     #yeh pehly B or M ko btay ga k total kitni bar aay ga

categories = {"B":1 , "M" : 2}


df["Label"] = df["Label"].replace(categories)    #Hum B or M ko sirf es liye change krain gy Q k hamara data Sirf Numeric values pr he kam krta ha JAB k B or M Char hain...

print(df["Label"].value_counts())     #B or M ko 1 or 2 sy change krny k bad tay ga k 1 or 2 total kitni bar aarha ha



#Labels
Y = df["Label"].values                   #es main hum ny label ki values ko Y k ander store krwaya jo k hum apny model k as a label dain gyy

#Features
X=df.drop(labels=["Label" , "id"] , axis=1)      #es main hum ny label wala column es liye drop kiya Q k wo hum as a label use kr rhy thy es liye usko features main use nai kr sakty thy  or id wala column es liye drop kiya Q k uski koi zarorat nai thi

print(X.head())


#Normalzation steps From Line 55 to 61 

from sklearn.preprocessing import MinMaxScaler        
from sklearn.preprocessing import QuantileTransformer

#Min Max Scaler hamari bht bari or choti values ko 0 sy 1 k beach main change kr deta ha jis sy predicton asan ho jati ha
scaler = MinMaxScaler()          #yeh min max scaler es liye use krty hain Q k hamary data main kch values bht zada bari hoti hain or kch values bht zada choti hain yeh dono values ko esy scale krta ha k hamari values main gap bht zada nai rehta jis ki waja graph easy fit ho jata ha
scaler.fit(X)    #jo min max scaler ka object banaya tha us k ander X(features ki values )  enter ki
X = scaler.transform(X)    #yeh transform ka function chnage kr deta ha 0 sy 1 tk values ko X ki values ko or duara usko X k ander store krwa diya



#MAchine learning steps
from sklearn.model_selection import train_test_split          #yeh hamary liye test data or trained data ko randomly alag kr dy ga jis sy hum apny model ko check kr sakty hain 



#X hamary features thy or Y hamary labels thy jo hum ny apny data sy alag kiye thy
#testing data 0.2 ka matlab 20 percent select kiya test k liye
#random seed hota ha matlab random state =42 ka matlab k yeh jo data randomly generate kr k test or train waly main store kry ga agr next time hum chahty hain k hamy yehi data dubara mily jub yeh function lagain to huamain yehi values random_state=42 likhna pry ga
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42)        


from sklearn import svm
model = svm.LinearSVC(max_iter=10000)         #SVM(Support vector machine)YEh model ha jo hamayr data ko fit krta ha matlab hamara model banata ha phr jub testing data dety hain to usko predict krta ha predict krta ha   iter=10000 ka matlab k 10000 bar chala chala kr data ko saii trah fit kry ga
model.fit(X_train , Y_train)  #model banay k bad us main features or Label ko fitttt kiya


prediction = model.predict(X_test)          #Jab hum ny apna model bana liya us k bad hum ny thora test data jo 20% tha jisko spliter k zariye alag kiya tha wo apny model k dy kr check krain gy k result thk aarhy hain k nai


#X test hamary featues hain jo hum apny data ko dety hain labels check krny k liye y test hamary labels hain 
from sklearn import metrics
print("Accuracy = ",metrics.accuracy_score(Y_test , prediction))     #jo prediction aai usko ko jo hamary labels thy us sy chcek krain gy k kitna percent correct ha



from sklearn.metrics import confusion_matrix        #confusion metrix ka matlab k yeh btata ha k hamary kitny labaels saii predct kye hamary model ny or kitny labels galat predict kiye
cm = confusion_matrix(Y_test , prediction)     #[[70  1]         #Total 72 B thy jin main sy 70 saii predict kiye or 2 galat  or 42 M thy jin main sy 41 saii predict kiye or 1 galat
                                               #[ 2 41]]
print(cm)

