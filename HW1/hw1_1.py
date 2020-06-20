""" This is the code for Question 1 Part 1
    __author__  = "Arda Andırın"
"""
import operator
import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def calculations(x, y, degree, color):
    # This function does all the necessary computations, written to avoid repetition.
    """
    :param x: x values
    :param y: y values
    :param degree: Polynomial degree
    :param color: Color of the line
    :return: No return instead plots the line.
    """
    model = LinearRegression()  # Our model
    poly = PolynomialFeatures(degree=degree)    # Polynomial degree function
    x_poly = poly.fit_transform(x)  # Depending on the degree x_poly takes that degree form of x.
    model.fit(x_poly, y)    # Train the model.
    y_pred = model.predict(x_poly)  # Prediction of y on the value x_poly

    # This part sorts the data for decent lines.
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x, y_pred), key=sort_axis)
    x_poly, y_pred = zip(*sorted_zip)

    label = ("Degree = " + str(degree))
    plt.plot(x_poly, y_pred, color=color, label=label)  # Draw the line


m = 100
x = np.random.rand(m, 1)*2
y = np.sin(2*math.pi*x) + np.random.randn(m, 1)

fig = plt.figure()
plt.scatter(x, y, s=30, color="b", label="Training points")
calculations(x, y, 0, color="r")
calculations(x, y, 1, color="g")
calculations(x, y, 3, color="m")
calculations(x, y, 9, color="y")


plt.xlabel("X")
plt.ylabel('Y')
plt.legend(loc="upper right")
plt.title("Case 3")
plt.show()