import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

#---------------- 1. DATA PREPROCESSING ---------------------------------

#import dataset - Data of customer of a bank
dataset = pd.read_csv("Dataset/Churn_Modelling.csv")
X = dataset.iloc[:,3:13].values  #3 to 12 column is taken, upper bound is ignored
Y = dataset.iloc[:,13].values

#encode categorical data
labelencoder_X1 = LabelEncoder()
X[:,2] = labelencoder_X1.fit_transform(X[:,2])
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [1])], remainder = 'passthrough')
X = transformer.fit_transform(X)

#drop 1 dummy variable to avoid falling into dummy variable trap
X = X[:, 1:]

#split dataset
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#Feature Scaling
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)

#-------------------- 2. BUILD ANN MODEL --------------------------------

#initialize ANN
classifier = Sequential()

#add input layer and hidden layer
#(units -> it is for hidden layer, input_dim = input layer)
#no. of nodes in hidden layer = avg of no. of nodes in input and output layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11)) 

#adding second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu')) 

#adding output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 

#compiling the ANN i.e. applying stochastic gradient descent or Adam
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fitting Ann to the training set
classifier.fit(train_x, train_y, batch_size = 10, nb_epoch = 100)

#------------- 3. MAKE PREDICTIONS AND EVALUATE MODEL -------------------

#Predicting the test results
y_pred = classifier.predict(test_x)
y_pred = (y_pred > 0.5)

#Performance evaluation using confusion matrix
cm = confusion_matrix(test_y, y_pred)

#compute accuracy
accuracy = (cm[0][0] + cm[1][1])/2000
