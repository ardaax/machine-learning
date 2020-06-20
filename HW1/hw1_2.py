"""
    CMPE442 Assignment 1, Part 2
    Using batch gradient descent on 100 points data.
    __author__  = "Arda Andırın"
"""
import numpy as np
import matplotlib.pyplot as plt

iterNo = 1000   # Number of iterations
eta = 0.1   # Learning rate
m = 100

theta = np.random.rand(2,1) # Initial random thetas.

# Generate our data
X=np.random.rand(m, 1)*2
y=100+3*X+np.random.randn(m, 1)


def linear_regression(X, y, iterNo, eta, theta):
    # Linear regression function that will return our thetas
    for i in range(iterNo):
        y_hat = np.dot(X, theta)    # y_hat is predicted y.
        gradient = np.dot(X.T, y_hat-y)
        theta = theta - eta * (1.0/m) * gradient    # Dot part is the summation of all the samples. This is where gradient descent finds the new theta.
    return theta


X_b = np.c_[np.ones((len(X), 1)), X]    # m x 2 matrix bias unit is added.
theta = linear_regression(X_b, y, iterNo, eta, theta)   # Find the thetas(list).
y_pred = np.dot(X_b, theta)     # the y we predict (y=θ theta x)

fig = plt.figure()
plt.scatter(X,y, s=40)    # Data points.
print(X)
print(y_pred)
plt.plot(X, y_pred, color='r', label="Line")  # Plot the gradient
plt.legend()
plt.title('Linear Regression with Batch Gradient Descent')
plt.show()