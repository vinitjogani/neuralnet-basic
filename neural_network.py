import numpy as np, pandas as pd

class NeuralNetwork:
    def __init__(self, layers = (1, 10, 1), alpha = .3, reg = 0, epochs=2, batch_size = 100):
        # Free parameters
        self.alpha = alpha
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size

        # Architecture
        self.layers = layers
        self.theta = [self.generate_theta(layers[i + 1], layers[i] + 1) for i in range(len(layers) - 1)]

    def generate_theta(self, rows, columns):
        """
            Randomly generate theta matrix.
        """
        return np.array([
            [(np.random.rand() - 0.5) for i in range(columns)] for j in range(rows)
        ])

    def g(self, z):
        """
            Sigmoid activation function.
        """
        overflow = lambda zi: 0 if -zi > np.log(np.finfo(np.float).max) else zi
        overflow = np.vectorize(overflow)
        return 1 / (1 + np.exp(-overflow(z)))

    def gprime(self, a):
        return 1 if a > 0 else 0

    def v(self, arr):
        """
            Column-vectorize an array.
        """
        return np.array(arr).reshape(len(arr), 1)

    def t(self, arr):
        """
            Row-vectorize an array.
        """
        return np.array(arr).reshape(1, len(arr))

    def transform_y(self, y):
        """
            Vectorize the labels.
        """
        return [[int(y[i] == j) for j in range(self.layers[-1])] for i in range(len(y))]

    def add_bias(self, x):
        """
            Adds a bias unit to activations of a layer.
        """
        return np.insert(x, 0, 1)

    def forward(self, x):
        """
            Forward propogation.
        """
        x = self.add_bias(x)
        activation = [x]

        for theta in self.theta:
            z = np.matmul(theta, self.v(x))
            x = self.add_bias(self.g(z))
            activation.append(x)
        return activation

    def backward(self, activation, y):
        """
            Backward propogation.
        """
        h = activation[-1][1:]

        delta = [[0 for j in range(i)] for i in self.layers]
        Delta = [[] for i in range(len(self.layers) - 1)]
        
        delta[-1] = (h - y) * (h) * (1 - h)
        for index in range(len(delta) - 1, 0, -1):
            i = index - 1
            delta[i] = np.matmul(self.theta[i].T[1:], delta[i + 1]) * activation[i][1:] * (1 - (activation[i][1:])) 
            Delta[i] = np.matmul(self.v(delta[i + 1]), self.t(activation[i]))

        return Delta

    def fit(self, X, y):
        """
            Fits the Neural Network's weights using backpropogation.
        """
        X = np.array(X)
        y = np.array(self.transform_y(y))

        for epoch in range(self.epochs):
            for batch in range(len(X) // self.batch_size):
                start_ix = batch * self.batch_size
                end_ix = (batch + 1) * self.batch_size

                start_cost = self.cost(X[start_ix:end_ix],  y[start_ix:end_ix])
                Delta = np.array(self.theta) * 0
                for i in range(start_ix, end_ix):
                    # Forward
                    activation = self.forward(X[i])

                    # Backward
                    DeltaNew = self.backward(activation, y[i])
                    for layer in range(len(self.theta)):
                        Delta[layer] = Delta[layer] + DeltaNew[layer]
                    
                # Update
                for layer in range(len(self.theta)):
                    reg_matrix = np.array(self.theta[layer]) * self.reg
                    reg_matrix.T[0] = 0
                    self.theta[layer] -= (self.alpha * Delta[layer] + reg_matrix) / self.batch_size

                end_cost = self.cost(X[start_ix:end_ix],  y[start_ix:end_ix])
                print(end_ix, start_cost, end_cost, end_cost - start_cost)

    def cost(self, X, y):
        """
            Calculate the Mean Squared Error.
        """
        sq_error = 0
        for i in range(len(X)):
            sq_error += sum((self.forward(X[i])[-1][1:] - y[i]) ** 2)
        return sq_error / len(X)

    def predict(self, X):
        """
            Predicts values for the inputs through forward propogation.
        """
        X = np.array(X)
        y = []
        for i in range(len(X)):
            y.append(np.argmax(self.forward(X[i])[-1][1:]))
        return y