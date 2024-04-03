import numpy as np


# ONE vs ALL
class SupportVectorMachine:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=100):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = {}

    def fit(self, X, y):
        n_classes = np.unique(y)

        for class_label in n_classes:
            binary_y = np.where(y == class_label, 1, -1)
            self.models[class_label] = self._binary_fit(X, binary_y)

    def _binary_fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, w) - b) >= 1
                if condition:
                    w -= self.lr * (2 * self.lambda_param * w)
                else:
                    w -= self.lr * (2 * self.lambda_param * w - np.dot(x_i, y[idx]))
                    b -= self.lr * y[idx]

        return (w, b)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.models)))

        for class_label, model in self.models.items():
            w, b = model
            approx = np.dot(X, w) - b
            predictions[:, class_label] = approx

        return np.argmax(predictions, axis=1)

