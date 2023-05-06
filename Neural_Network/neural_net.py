import numpy as np


class Neural_Net:

    def __init__(self):
        self.parameters = {}
        self.grads = {}

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def initialize_parameters(self, n_x, n_y):
        W = np.random.randn(n_y, n_x) * 0.01
        b = np.zeros((n_y, 1))

        self.parameters = {"W": W,
                           "b": b}

    def forward_propagation(self, X):
        W = self.parameters["W"]
        b = self.parameters["b"]

        Z = np.matmul(W, X) + b
        A = self.sigmoid(Z)

        return A

    def compute_cost(self, A, Y):
        # Number of examples.
        m = Y.shape[1]

        # Compute the cost function.
        logprobs = - np.multiply(np.log(A), Y) - \
            np.multiply(np.log(1 - A), 1 - Y)
        cost = 1/m * np.sum(logprobs)

        return cost

    def backward_propagation(self, A, X, Y):

        m = X.shape[1]

        dZ = A - Y
        dW = 1/m * np.dot(dZ, X.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)

        self.grads = {"dW": dW,
                      "db": db}

    def update_parameters(self, learning_rate=1.2):

        W = self.parameters["W"]
        b = self.parameters["b"]

        dW = self.grads["dW"]
        db = self.grads["db"]

        W = W - learning_rate * dW
        b = b - learning_rate * db

        self.parameters = {"W": W,
                           "b": b}

    def nn_model(self, X, Y, num_iterations=10, learning_rate=1.2):

        n_x = X.shape[0]
        n_y = Y.shape[0]

        self.initialize_parameters(n_x, n_y)

        for i in range(0, num_iterations):

            A = self.forward_propagation(X)

            cost = self.compute_cost(A, Y)

            self.backward_propagation(A, X, Y)

            self.update_parameters(learning_rate)

            print("Cost after iteration %i: %f" % (i, cost))

    def predict(self, X):

        A = self.forward_propagation(X)
        predictions = A > 0.5

        return predictions
    
