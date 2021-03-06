#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:32:07 2019

python 3.7
@author: sinamahbobi
"""

import numpy as np 

import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv') 
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values




#take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])

#encoding categorical data
#encode the independent variable

"""from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(),[0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float) 

#encoding Y Data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y) """

#splitting data set into the trainign set and data set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


