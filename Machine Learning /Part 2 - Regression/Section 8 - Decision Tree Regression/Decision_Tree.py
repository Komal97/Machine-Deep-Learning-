import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing dataset
dataset = pd.read_csv("Dataset/Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting Decision Tree Regression to dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) #Paramaters -> criterion-default="mse" -> mse = mean square error
regressor.fit(X, Y)

#Predicting the new result
y_pred = regressor.predict(np.array([[6.5]]))

#Visualizing the Decision Tree Regression results (for high resolution and smooth curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()