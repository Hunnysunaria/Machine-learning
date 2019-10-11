# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:09:58 2019

@author: Hunny Sunaria
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing datasets
dataset= pd.read_csv('Position_Salaries.csv')
X= dataset.iloc[:,1:2].values
Y= dataset.iloc[:,2].values

#splitting data for training and testing
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X,Y)
  #polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
X_Reg =  poly_reg.fit_transform(X)
poly_reg.fit(X_Reg,Y)
lin_reg2= LinearRegression()
lin_reg2.fit(X_Reg,Y)

#Visualizing the Linear regressiong
plt.scatter(X,Y,color='red')
plt.plot(X,linreg.predict(X),color='blue')
plt.title("Truth or Bluff(LR) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualizing the Polynomial regressiong
X_grid= np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Truth or Bluff(PR) ")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#prdict linear regression
linreg.predict(6.5)
#predict polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))
