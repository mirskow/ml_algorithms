import numpy as np


class NeuralNetwork:
    def __init__(self, layer, neurons, lr=0.01, era=100):
        self.layer = layer
        self.neurons = neurons
        self.weights = []
        self.bias = []
        self.lr = lr
        self.era = era
        self.input_dim = None
        self.out_dim = None
        self.loss_arr = []

    def _forward(self, x):
        activations = [np.array(x).reshape(1, len(x))]
        pre_activations = []
        for i in range(self.layer):
            pre_activation = activations[-1] @ self.weights[i] + self.bias[i]
            pre_activations.append(pre_activation)
            if i == self.layer - 1:
                activations.append(self._softmax(pre_activation))
            else:
                activations.append(self._activation(pre_activation))
        return activations, pre_activations

    def _backward(self, y, activations, pre_activations):
        dWs = []
        dbs = []
        dE_dt = activations[-1] - self._to_full(y, activations[-1].shape[1])
        for i in range(self.layer - 1, -1, -1):
            if i != self.layer - 1:
                dE_dh = dE_dt @ self.weights[i+1].T
                dE_dt = dE_dh * self._activation_deriv(pre_activations[i])
            dE_dW = activations[i].T @ dE_dt
            dE_db = dE_dt
            dWs.append(dE_dW)
            dbs.append(dE_db)
        return dWs, dbs

    def _update_weights(self, dWs, dbs):
        for i in range(len(self.weights)-1, -1, -1):
            self.weights[i] -= self.lr * dWs[i]
            self.bias[i] -= self.lr * dbs[i]

    def _to_full(self, y, out_dim):
        y_full = np.zeros((1, out_dim))
        y_full[0, y] = 1
        return y_full

    def _sparse_cross_entropy(self, z, y):
        if z[0, y] != 0:
            return -np.log(z[0, y])
        else:
            return -np.log(1e-15)

    def _activation(self, t):
        return np.maximum(t, 0)

    def _activation_deriv(self, t):
        return (t >= 0).astype(float)

    def _softmax(self, t):
        out = np.exp(t - np.max(t))
        return out / np.sum(out)

    def fit(self, X_train, y_train):
        self.input_dim = int(X_train.shape[1])
        self.out_dim = len(np.unique(y_train))

        #initialize weight and bias
        for i in range(self.layer):
            if i == 0:
                weightn = np.random.randn(self.input_dim, self.neurons)
                biasn = np.random.randn(1, self.neurons)
            elif i+1 == self.layer:
                weightn = np.random.randn(self.neurons, self.out_dim)
                biasn = np.random.randn(1, self.out_dim)
            else:
                weightn = np.random.randn(self.neurons, self.neurons)
                biasn = np.random.randn(1, self.neurons)

            self.weights.append(weightn)
            self.bias.append(biasn)

        #education
        for e in range(self.era):
            for i in range(len(X_train)):
                x = X_train[i]
                y = y_train[i]
                #forward
                activations, pre_activations = self._forward(x)
                #mse
                E = self._sparse_cross_entropy(activations[-1], y)
                self.loss_arr.append(E)
                #backward
                dWs, dBs = self._backward(y, activations, pre_activations)
                #update parametrs
                self._update_weights(dWs[::-1], dBs[::-1])

    def predict(self, X_test):
        probabilities = []
        for i in range(len(X_test)):
            prediction, _ = self._forward(X_test[i])
            probabilities.append(prediction[-1])
        predictions = []
        for probability in probabilities:
            predictions.append(np.argmax(probability[0]))
        return predictions
