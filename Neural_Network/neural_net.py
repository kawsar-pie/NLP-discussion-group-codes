import numpy as np


class Neural_Net:
    def __init__(self):
        self.parameters = {"W": 0, "b": 0}
        self.grads = {}

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def initialize_parameters(self, no_samples, no_features):
        self.parameters["W"] = np.zeros((no_features, 1))
        self.parameters["b"] = 0
        # print("initialize_parameters")
        # print(self.parameters["W"].shape)

        # self.parameters = {"W": W,
        #                    "b": b}

    def forward_propagation(self, X):
        W = self.parameters["W"]
        b = self.parameters["b"]
        # print("forward_propagation")
        # print(X.shape, W.shape)
        Z = np.matmul(X, W) + b
        A = self.sigmoid(Z)

        return A

    def compute_cost(self, A, Y):
        cost = np.square(np.subtract(A, Y)).mean()
        return cost

    def backward_propagation(self, A, X, Y):
        m = X.shape[0]
        Y = Y.reshape(m, 1)
        dZ = A - Y
        dW = 2/m * np.dot(X.T, dZ)
        db = 2/m * np.sum(dZ, axis=0, keepdims=True)

        # Print shapes for debugging
        # print("backward_propagation")
        # print("dW", dW.shape)
        # print("X", X.shape)
        # print("dZ", dZ.shape)
        # print("A", A.shape)
        # print("Y", Y.shape)
        # print("db", db.shape)

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
        no_features = X.shape[1]
        no_samples = X.shape[0]
        Y = Y.reshape(no_samples, 1)
        self.initialize_parameters(no_samples, no_features)

        for i in range(num_iterations):
            A = self.forward_propagation(X)
            cost = self.compute_cost(A, Y)
            self.backward_propagation(A, X, Y)
            self.update_parameters(learning_rate)
            print("Cost after iteration %i: %f" % (i, cost))

    def predict(self, X):
        A = self.forward_propagation(X)
        predictions = (A > 0.5).astype(int)
        return predictions
