from neural_net import Neural_Net
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=10000)
# print("X Y")
# print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42, test_size=0.2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )
dnn = Neural_Net()
dnn.nn_model(X_train, y_train, num_iterations=100, learning_rate=0.0001)

y_pred = dnn.predict(X_test)
y_pred =  y_pred.reshape(y_pred.shape[0],)
# print(X_test)
# print(y_test)
# print(y_pred.shape, y_test.shape)
print(accuracy_score(y_test, y_pred))
