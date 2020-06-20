""" NN IMPLEMENTATION PROVIDED BY THE INSTRUCTOR
    USED IN QUESTION 7
"""

import numpy as np
import random

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 - z)


class NeuralNetwork:
    def __init__(self, inSize, sl2, clsSize, lrt):

        self.iSz = inSize
        self.oSz = clsSize
        self.hSz = sl2
        self.weights1 = (np.random.rand(self.hSz, self.iSz + 1) - 0.5) / np.sqrt(self.iSz)
        self.weights2 = (np.random.rand(self.oSz, self.hSz + 1) - 0.5) / np.sqrt(self.hSz)

        self.output = 0
        self.layer1 = np.zeros(self.hSz)
        self.eta = lrt

    def feedforward(self, x):
        x_c = np.r_[1, x]
        self.layer1 = sigmoid(np.dot(self.weights1, x_c))
        layer1_c = np.r_[1, self.layer1]
        self.output = sigmoid(np.dot(self.weights2, layer1_c))

    def backprop(self, x, trg):

        sigma_3 = (trg - self.output)  # outer layer error
        sigma_3 = np.reshape(sigma_3, (self.oSz, 1))

        layer1_c = np.r_[1, self.layer1]  # hidden layer activations+bias
        sigma_2 = np.dot(self.weights2.T, sigma_3)
        tmp = sigmoid_derivative(layer1_c)
        tmp = np.reshape(tmp, (self.hSz + 1, 1))
        sigma_2 = np.multiply(sigma_2, tmp)  # hidden layer error
        delta2 = sigma_3 * layer1_c  # weights2 update

        x_c = np.r_[1, x]  # input layer +bias
        delta1 = sigma_2[1:, ] * x_c  # weights1 update

        return delta1, delta2

    def fit(self, X, y, iterNo):

        m = np.shape(X)[0]
        for i in range(iterNo):
            D1 = np.zeros(np.shape(self.weights1))
            D2 = np.zeros(np.shape(self.weights2))
            for j in range(m):
                self.feedforward(X[j])
                [delta1, delta2] = self.backprop(X[j], y[j])
                D1 = D1 + delta1
                D2 = D2 + delta2
            self.weights1 = self.weights1 + self.eta * (D1 / m)
            self.weights2 = self.weights2 + self.eta * (D2 / m)

    def predict(self, X):

        m = np.shape(X)[0]
        y = np.zeros(shape=(m,self.oSz))
        for i in range(m):
            self.feedforward(X[i])
            y[i] = self.output
        return y
