import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Dataset/Salary_Data.csv")
#-1 means calculate the dimension of rows, but have 1 column
X = dataset.iloc[:,0].values.reshape(-1, 1)
Y = dataset.iloc[:,-1].values.reshape(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set results
#y_pred is the vector of the predictions of dependent variables
#here it holds predicted salaries as it is dependent variable
y_pred = regressor.predict(X_test)

#Visualizing the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Yrs of Experience')
plt.ylabel('Salary ($)')
plt.show()

#Visualizing the training set results
plt.scatter(X_test, Y_test, color='magenta')
plt.plot(X_test, y_pred, color='green')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Yrs of Experience')
plt.ylabel('Salary ($)')
plt.show()