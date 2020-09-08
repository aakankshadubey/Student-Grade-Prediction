#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:23:25 2020
Student Grade Prediction (Using Linear Regression)
@author: Aakanksha Dubey
"""

# Importing Libraries
import pandas as pd
import numpy as np


# Importing Dataset
dataset = pd.read_csv("student-mat.csv", sep = ';')
dataset = dataset[["studytime", "absences", "failures", "G1", "G2", "G3"]] 
X = dataset.iloc[:, : -1].values    
y = dataset.iloc[:, -1].values


# Splitting Data into Training Set and Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0) 


# Fit Linear Regression Model to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting Values of Test Set
y_pred = regressor.predict(X_test)
accuracy = regressor.score(X_test, y_test)
print("Accuracy : {:.8f}".format(accuracy))


# Coeffecients And Intercept of Linear Regression Line
print("Coeffecients :  {}".format(regressor.coef_))
print("Intercept :  {}".format(regressor.intercept_))

