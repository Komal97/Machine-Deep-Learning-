import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#A new employee tell his salary in prv company as 160K
#Hr confirms from prv employer and got a sheet with the provided dataset
#Hr has to identify and confirm the salary of new employee from prv company data
dataset = pd.read_csv("Dataset/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
simple_linear_regressor = LinearRegression()
simple_linear_regressor.fit(X,Y)

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures #PolynomialFeaturescontains function for introducing polynomial terms
poly_regressor = PolynomialFeatures(degree = 4) #this obj -> transformation tool -> for transforming X into new with polynomial X values
X_poly = poly_regressor.fit_transform(X)
linear_regressor = LinearRegression()
linear_regressor.fit(X_poly, Y)

#Visualizing the Linear Regression results
plt.scatter(X, Y, color = 'red')
plt.plot(X, simple_linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression results
# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape(len(X_grid),1)
# plt.scatter(X, Y, color = 'red')
# plt.plot(X_grid, linear_regressor.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.scatter(X, Y, color = 'red')
plt.plot(X, linear_regressor.predict(poly_regressor.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting new result with Linear Regression
simple_linear_regressor.predict(np.array([[6.5]]))

#Predicting new result with Polynomial Regression
linear_regressor.predict(poly_regressor.fit_transform(np.array([[6.5]])))