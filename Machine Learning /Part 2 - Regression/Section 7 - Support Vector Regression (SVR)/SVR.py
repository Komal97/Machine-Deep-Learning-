import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("Dataset/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Feature Scaling

#here it is included because this is not done in SVR class itself but present in linear regression
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1,1))

#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, Y.flatten())

#Predicting the new result
a = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_Y.inverse_transform(a)

#Visualizing the SVR Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

