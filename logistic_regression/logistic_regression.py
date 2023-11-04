# logistic_regression.py
import numpy as np

class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def initialize_parameters(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_cost_and_gradient(self, X, Y):
        m = X.shape[1]
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        dw = 1/m * np.dot(X, (A - Y).T)
        db = 1/m * np.sum(A - Y)
        return cost, dw, db

    def optimize(self, X, Y, num_iterations, learning_rate):
        for i in range(num_iterations):
            cost, dw, db = self.compute_cost_and_gradient(X, Y)
            self.w = self.w - learning_rate * dw
            self.b = self.b - learning_rate * db

    def predict(self, X):
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        for i in range(A.shape[1]):
            Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        return Y_prediction
