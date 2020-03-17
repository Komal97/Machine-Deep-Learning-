import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

#---------------- 1. DATA PREPROCESSING ---------------------------------

#import dataset - Data of customer of a bank
dataset = pd.read_csv("dataset/Churn_Modelling.csv")
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

#-------------------- 2. BUILD ANN MODEL --------------------------------
classifier = XGBClassifier()
classifier.fit(train_x, train_y)

#------------- 3. MAKE PREDICTIONS AND EVALUATE MODEL -------------------

#Predicting the test results
y_pred = classifier.predict(test_x)

#1st method - Performance evaluation using confusion matrix
cm = confusion_matrix(test_y, y_pred)

#compute accuracy
accuracy = (cm[0][0] + cm[1][1])/2000

#2nd method - Performance evaluation using k-fold cross validation
accuracies = cross_val_score(estimator = classifier, X = train_x, y = train_y, cv = 10) # cv = k-ƒold value
accuracies.mean()
accuracies.std()


