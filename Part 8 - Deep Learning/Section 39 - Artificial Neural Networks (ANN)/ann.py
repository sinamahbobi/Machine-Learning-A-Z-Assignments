# Artificial Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encode categorical data 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder 

labelencoder_X1 = LabelEncoder()
X[:,1] = labelencoder_X1.fit_transform(X [:, 1])
labelencoder_X2 = LabelEncoder()
X[:,2] = labelencoder_X2.fit_transform(X [:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# splitting data into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing the ANN
classifier = Sequential()

# adding the input layer and first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# add 2nd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# add output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# compile the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 Making Predictions and evaluating the model

y_pred = classifier.predict(X_test)
y_pred = (y_pred> 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




