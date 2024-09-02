import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
print("X and Y")
print(X, "\n\n", Y.reshape(-1,1), "\n")

#Splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
print("X train and test")
print(X_train, "\n\n", X_test, "\n")
print("Y train and test")
print(Y_train.reshape(-1,1), "\n\n", Y_test.reshape(-1,1), "\n")

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lars
from sklearn.ensemble import AdaBoostRegressor
regressor = LinearRegression()
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(regressor,"\n")

#Predicting the test set results
# Y_pred = regressor.predict(X_test)
Y_pred = regressor.coef_*(X_test) + regressor.intercept_

print("Y prediction","\n")
print(Y_pred.reshape(-1,1),"\n")

#Visualizing the Training Set Results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Salary vs Experience (Training Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the Test Set Results
plt.scatter(X_test, Y_test, color='green')
plt.plot(X_test, Y_pred, color='black')
plt.title("Salary vs Experience (Test Set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()


# from sklearn.linear_model import LinearRegression
# #Fitting Polynomial Regression to dataset
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree=14)
# X_poly = poly_reg.fit_transform(X)
# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(X_poly, Y)

# #Visualizing the results
# plt.scatter(X, Y, color="red")
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color="blue")
# plt.title("Salary vs Experience (Test Set)")
# plt.xlabel("Years of experience")
# plt.ylabel("Salary")
# plt.show()

