import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, inSize, sl2, clsSize, lrt):

        self.iSz = inSize  # number of input units
        self.oSz = clsSize  # number of output units
        self.hSz = sl2  # number of hidden units
        # initialize weights
        self.weights1 = (np.random.rand(self.hSz, self.iSz+1) - 0.5) / np.sqrt(self.iSz)  # Weights of input to hidden
        self.weights2 = (np.random.rand(self.oSz, self.hSz+1) - 0.5) / np.sqrt(self.hSz)  # Weights of hidden to output
        self.eta = lrt  # learning rate
        # Other stuff you think you are going to need
        self.output = 0
        # This holds the needed data for backwards propagation
        self.data = {"Z1": 0,
                     "A1": 0,
                     "Z2": 0,
                     "A2": 0}


    def sigmoid(self, x):
        return 1/(1+np.exp(-x))


    def feedforward(self, x):
        # is computed for single training sample
        """ x is of shape (1,iSz) so we take transpose"""
        # Compute the activation of each neuron j in the hidden layer  𝑎^((𝑙))
        z1 = (np.dot(self.weights1, x.T))   # This is z^[1] and shape= (hSz, 1)
        a1 = self.sigmoid(z1)
        a1 = np.append(a1, [1]).reshape(self.hSz+1,1)   # Adding bias element
        # a1 = np.array([a1[0], a1[1], [1]])
        z2 = np.dot(self.weights2, a1)
        a2 = self.sigmoid(z2)

        # Update values to be used in backwards propagation
        self.data["Z1"] = z1
        self.data["A1"] = a1
        self.data["Z2"] = z2
        self.data["A2"] = a2

        self.output = a2
        # Until the output unit is reached

    def backprop(self, x, trg):
        # is computed for single training sample

        # Compute the error at the output  𝛿^((𝐿))= trg-𝑎^((3))   --- Here target-output, not other way around
        # Compute 𝛿^(2). Look into slides
        # Compute the derivarive of cost finction with respect to
        # each weight in the network  𝜕/(𝜕〖𝜃_𝑖𝑗〗^((𝑙) ) ) 𝐽(𝜃)=〖𝑎_𝑗〗^((𝑙)) 〖𝛿_𝑖〗^((𝑙+1))
        dz2 = trg - self.data["A2"]
        dw2 = np.dot(dz2, self.data["A1"].T)
        # db2 = dz2

        dz1 = np.dot(self.weights2.T, dz2)*((1-self.data["A1"])*self.data["A1"])
        dz1 = dz1[:-1]  # Remove bias element

        dw1 = np.dot(dz1, x)
        #db1 = dz1


        dc2 = np.dot(self.data["A2"], dz2.T)
        dc1 = np.dot(self.data["A2"], dz1.T)
        return [dw1, dw2]


    def fit(self, X, y, iterNo):
        m = np.shape(X)[0]  # Number of training examples
        cost_list = list()
        iter_list = list()
        for i in range(iterNo):
            D1 = np.zeros(np.shape(self.weights1))  # Empty init of D1
            D2 = np.zeros(np.shape(self.weights2))  # Empty init of D2
            for j in range(m):
                self.feedforward(X[j].reshape(1,self.iSz+1))
                [delta1, delta2] = self.backprop(X[j].reshape(1,self.iSz+1), y[j])
                D1 = D1 + delta1
                D2 = D2 + delta2
            self.weights1 = self.weights1 + self.eta * (D1 / m)  # Update weights1
            self.weights2 = self.weights2 + self.eta * (D2 / m)  # Update weights2
            # Compute error function after each 100 iterations
            y_pred = 0

            if i == iterNo-1:
                y_pred = self.predict(X)
                print("Final prediction is", y_pred)
            if ((i % 100 == 0) or (i == iterNo-1)):
                cost = self.cal_cost(X,y)
                cost_list.append(cost)
                iter_list.append(i)
                print("Iteration number: %d | Error = %.5f" % (i, cost[0]))
        plt.scatter(iter_list, cost_list)
        plt.plot(iter_list, cost_list)
        plt.xlabel("Iteration number")
        plt.ylabel("Cost")
        plt.title("Error at each 100 iterations, hidden size = %d" % self.hSz)
        plt.show()

    def cal_cost(self, X, y):
        y_pred = self.predict(X)
        total = 0
        for x in range(np.shape(X)[0]):
            total += np.square(y_pred[x] - y[x])
        cost = total
        return cost


    def predict(self, X):

        m = np.shape(X)[0]  # Number of training exapmles
        y = np.zeros(m)  # Filled with zeros
        for i in range(m):
            self.feedforward(X[i].reshape(1,self.iSz+1))  # For every input predict
            y[i] = self.output
        return y


def main():
    inp = 2  # Number of units in input layer
    hidden = 2  # Number of units in hidden layer
    output = 1  # Number of units in output layer
    lrt = 1
    nn = NeuralNetwork(inp, hidden, output, lrt)
    """ x1 is inputs for AND function"""
    x1 = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])
    y1 = np.array([[0],
                   [0],
                   [0],
                   [1]])

    # nn.fit(x1,y1, iterNo=500)

    """ x2 is inputs fcor XOR function"""
    x2 = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])
    y2 = np.array([[0],
                   [1],
                   [1],
                   [0]])

    nn.fit(x2, y2, iterNo=1500)

if __name__ == '__main__':
    main()