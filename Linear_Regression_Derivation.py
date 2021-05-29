#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[58]:


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.normal_eq = None

    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for iters in range(self.n_iters):
            
            # Prediction
            y_pred = np.dot(X, self.w) + self.b
            
            # Gradient Descent
            dw = (1/n_samples) * np.dot(X.T,(y_pred - y)) 
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Updating parameters
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            

    def predict(self, X):
        y_approximated = np.dot(X, self.w) + self.b
        return y_approximated
    
    
    def fit_OLS(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X = np.append(ones, X, axis = 1)
        Xt_X_inv = np.linalg.pinv(np.dot(X.T, X))
        Xt_y = np.dot(X.T, y)
        self.normal_eq = np.dot(Xt_X_inv, Xt_y)
        return self.normal_eq
        
    def predict_OLS(self, X):
        y_approximated = np.dot(X, self.normal_eq[1]) + self.normal_eq[0]
        return y_approximated


# In[67]:


X, y = datasets.make_regression(n_samples=1000, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# In[69]:


# Fitting the regression line
regressor = LinearRegression()
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

# This uses OLS
regressor.fit_OLS(X_train, y_train)
predicted_OLS = regressor.predict_OLS(X_test)


# In[83]:


plt.figure(figsize =(8,6))
plt.plot(X_test, predicted)
plt.plot(X_test, predicted_OLS, color = 'r', label='OLS method')
plt.scatter(X_train, y_train, color = 'orange', label='Gradient Descent Method', alpha =0.5)
plt.legend()
plt.show()


# In[ ]:




