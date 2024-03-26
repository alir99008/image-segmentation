# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 11:39:18 2022

@author: Hp
"""

import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()   #es main hum ny jo data set liya tha usko housing k avaribale k ander store krwa diya

df = pd.DataFrame(data= np.c_[housing['data'], housing['target']],            #data ka matlab ho ha sirf data wali dictionary k ander jitny rows or columns hain un sub ka data means values ly ga or target ki dictinoary k andr jitni values hain unka B
                     columns= housing['feature_names'] + ['target'])     #column main sirf name leta ha jo k data jo uper liya tha dictonaries sy us k columns name bun jaty hain or yeh hum features wali dictionariy sy lain gy or 1 target wala column B laingy



#df = pd.read_csv("data/normalization.csv")

print(df.describe().T)


#Define the dependent variable that needs to be predicted (labels)
Y = df["target"].values           #es main hum ny target ki values ko Y k ander store krwaya jo k hum apny model k as a label dain gyy

#Define the independent variables. Let's also drop Gender, so we can normalize other data
X = df.drop(labels = ["target"], axis=1)     #es main hum ny target wala column es liye drop kiya Q k wo hum as a label use kr rhy thy es liye usko features main use nai kr sakty thy  


sns.distplot(df['MedInc'], kde=False)   #yeh MedInc waly column ka graph show kr dy ga agr kde false Q kiya es ki damj nai aarhi to true kr k dekh lain samj aajay gi
sns.distplot(df['AveOccup'], kde=False) # Large Outliers. 1243 occupants?
sns.distplot(df['Population'], kde=False) #Outliers. 35682 max but mean 1425

X = X[['MedInc', 'AveOccup']].copy()    #es main hamary data main sy jo X main store krwaya tha us main sy sirf 2 column or uska X k ander rewrite kr dy ga
column_names = X.columns    #es main sirf column k name column name main store kry ga

sns.jointplot(x='MedInc', y='AveOccup', data=X, xlim=[0,10], ylim=[0,5] ) # xlim=[0,10], ylim=[0,5]

###################################################################################
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

#Other transformations not shown below. 
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer

##############################################################################
#1 Standard scaler
#removes the mean and scales the data to unit variance.
# But, outliers have influence when computing mean and std. dev.
scaler1 = StandardScaler()
scaler1.fit(X)
X1 = scaler1.transform(X)
df1 = pd.DataFrame(data=X1, columns=column_names)
print(df1.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df1)  #Data scaled but outliers still exist


#2 MinMaxScaler
#rescales the data set such that all feature values are in the range [0, 1] 
#For large outliers, it compresses lower values to too small numbers.
#Sensitive to outliers.
scaler2 = MinMaxScaler()
scaler2.fit(X)
X2 = scaler2.transform(X)
df2 = pd.DataFrame(data=X2, columns=column_names)
print(df2.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df2, xlim=[0,1], ylim=[0,0.005])  #Data scaled but outliers still exist

#3 RobustScaler
# the centering and scaling statistics of this scaler are based on percentiles 
#and are therefore not influenced by a few number of very large marginal outliers.
scaler3 = RobustScaler()
scaler3.fit(X)
X3 = scaler3.transform(X)
df3 = pd.DataFrame(data=X3, columns=column_names)
print(df3.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df3, xlim=[-2,3], ylim = [-2,3]) #Range -2 to 3


#4 PowerTransformer
# applies a power transformation to each feature to make the data more Gaussian-like
scaler4 = PowerTransformer()
scaler4.fit(X)
X4 = scaler4.transform(X)
df4 = pd.DataFrame(data=X4, columns=column_names)
print(df4.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df4) #

#5 QuantileTransformer
# has an additional output_distribution parameter allowing to match a 
# Gaussian distribution instead of a uniform distribution.
scaler5 = QuantileTransformer()
scaler5.fit(X)
X5 = scaler5.transform(X)
df5 = pd.DataFrame(data=X5, columns=column_names)
print(df5.describe())
sns.jointplot(x='MedInc', y='AveOccup', data=df5) #