"""
    CMPE442 Assignment 1, Part 3
    Using batch gradient descent on 100 points data.
    __author__  = "Arda Andırın"
"""
from scipy import linalg
import operator
import numpy as np
import matplotlib.pyplot as plt
import math
# mse = np.zeros((2,)*1)  # 1 x 2 matrix,


def weighted_linear_regression(X, y, iteration_cnt, eta, x, tau, theta):
    weights = np.array([np.exp(- ((i - X[x]) ** 2) / (2 * (tau ** 2))) for i in range(m)])  # All the weights
    for iteration in range(iteration_cnt):
        ''' Commented part is what I tried first.'''
        # y_hat = np.dot(X, theta)  # y_hat is predicted y.
        # gradient = np.dot(X.T, y_hat - y)  #
        # gradient = np.dot(gradient.T, w.T)
        b = np.array([np.sum(weights * y), np.sum(weights * y * X)])    # For theta0
        A = np.array([[np.sum(weights), np.sum(weights * X)],           # For theta1
                      [np.sum(weights * X), np.sum(weights * X * X)]])
        theta = linalg.solve(A, b)
        theta = theta - eta * (1.0 / m)     # Gradient descent
        # theta = theta - eta * (1.0/m) * np.dot(X.T, y_hat-y)  # Dot part is the summation of all the samples. This is where gradient descent finds the new theta.
    return theta


theta = np.random.rand(2,1) # Initial random thetas.
m = 100
iterNo = 100   # Number of iterations
eta = 0.4   # Learning rate
tau = 1   # Bandwidth parameter
theta_list = list()
y_pred_list = np.zeros(m)
y_pred_list = y_pred_list.reshape((m,1))
X = np.random.rand(m, 1)*2
y = np.sin(2*math.pi*X)+np.random.randn(m, 1)
X_b = np.c_[np.ones((len(X), 1)), X]    # m x 2 matrix, bias unit is added.
# w = np.array([np.exp(- (i - X_b[i])**2/(2*(tau**2))) for i in range(m)])    # All the weights

for i in range(m):
    theta = weighted_linear_regression(X_b, y, iterNo, eta, i, tau, theta)  # Find the thetas(list).
    theta_list.append(theta)
    y_pred = X_b[i][0]*theta[0] + X_b[i][1] * theta[1]
    y_pred_list[i] = y_pred
# y_pred = np.dot(X_b, theta)     # the y we predict (y=θ theta x)
# print(y_pred_list)

# This part sorts the data for decent lines.
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X, y_pred_list), key=sort_axis)
X, y_pred_list = zip(*sorted_zip)



fig = plt.figure()
plt.scatter(X,y, s=40)    # Data points.
plt.plot(X, y_pred_list, color='r', label="Line")  # Plot the gradient
plt.legend()
plt.title('Locally Weighted Linear Regression')
plt.show()