import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500)  # Prevent overflow encountering large values
    return 1 / (1 + np.exp(-x))

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_prediction = np.dot(X, self.weights) + self.bias
            prediction = sigmoid(linear_prediction)

            dw = (1 / n_samples) * np.dot(X.T, (prediction - y))
            db = (1 / n_samples) * np.sum(prediction - y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_prediction = np.dot(X, self.weights) + self.bias
        y_prediction = sigmoid(linear_prediction)
        y_predicted_cls = [1 if y > 0.5 else 0 for y in y_prediction]
        return y_predicted_cls