import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Dataset/50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#Encoding the categorial data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [3])], remainder = 'passthrough')
X = transformer.fit_transform(X)

#Avoiding the Dummy Variable Trap
X = X[:,1:]  #start from index 1 to n and ignore 0 index

#Creating training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm  #sm = stats model

#sm doesn't consider b0 constant
#so we have to add a column with value = 1, considering x0 = 1 so that sm will consider b0
#axis = 1 -> column
#X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis = 1)  #here column ones is added to arr X at the end
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)   #here array X is added to arr of ones
#creating optimal matrix of features
#X_opt contains only independent variables that are highly statistically significant for calulating dependent variable
X_opt = X[:,[0,1,2,3,4,5]] 
#ols = ordinary least square
regressor_OLS =  sm.ols(endog = Y, exog = X_opt, data = dataset, formula = 'R&D Spend + Administration + Marketing Spend + State').fit()
#contain information about multilinear regression model
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]  # 2 has highest P value so removing it
regressor_OLS = sm.OLS(Y, X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]  
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]  
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]  
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()