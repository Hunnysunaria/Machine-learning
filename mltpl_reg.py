# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:34:17 2019

@author: Hunnysunaria
"""


"""#importing libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""#importing datasets"""
dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


"""#categorize data"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
categories='auto'

labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

"""#Avoiding dummy variable trap"""
x=x[:,1:]

"""#splitting data into training and testing"""

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

"""#linear model"""
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

"""#predicting testset result"""
y_pred=regressor.predict(X_test)

"""#building a optimal model using backward elimination"""
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
            