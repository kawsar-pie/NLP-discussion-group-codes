from neural_net import Neural_Net
import numpy as np

X = np.random.randint(0, 2, (2, 30))
Y = np.logical_and(X[0] == 0, X[1] == 1).astype(int).reshape((1, 30))
dnn = Neural_Net()
dnn.nn_model(X, Y, num_iterations=50, learning_rate=1.2)