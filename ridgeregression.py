# -*- coding: utf-8 -*-
"""RidgeRegression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BKksb56YaSS2wIPFZGWUmI02qf6oUdYW
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import datasets

# We can implement ridge regression using gradient descent too.
# This uses the closed method
class RidgeRegression:

  def __init__(self, lamda=1):
    self.lamda = lamda  #This quantitatively decides which features should be important
  
  def fit(self, X, y):
    # Adding he intercepts
    ones = np.ones((X.shape[0], 1))
    self.X = np.concatenate((ones, X), axis=1)

    #Creating the penalty matrix
    penalty_matrix = np.diag(np.ones(self.X.shape[1]))
    penalty_matrix[0, 0] = 0  # The first element should be a zero

    #Below are the parts of the equation
    X_t_X = np.dot(self.X.T, self.X)
    lamda_penalty_matrix = self.lamda * penalty_matrix
    #The shape of the penalty matrix should (n_features + 1 x n_features + 1)
    X_t_y = np.dot(self.X.T, y)

    #Final part of the equation
    self.thetas = np.dot(np.linalg.inv(X_t_X + lamda_penalty_matrix), X_t_y)
    return self.thetas

  def predict(self, X):
    thetas = self.thetas
    ones = np.ones((X.shape[0], 1))
    X_predictor = np.concatenate((ones, X), axis = 1)
    self.y_approximations = np.dot(X_predictor, thetas)
    return self.y_approximations


# Testing the code
X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# This has only one features, Ridge Regression would just act like a LinearRegression

ridge_reg = RidgeRegression(lamda = 1) # Lambda here doesn't have much of an effect
# use Ridge Regression if we have more features
ridge_reg.fit(X_train, y_train)
y_predicted = ridge_reg.predict(X_test)

# Visualize the prediction and actualy values
plt.scatter(X_test, y_test, c='orange', alpha = 0.5)
plt.plot(X_test, y_predicted, c='black')
plt.show()
# We have good fit

