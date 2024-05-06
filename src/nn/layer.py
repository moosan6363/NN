import numpy as np


class Layer:
    def __init__(self):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError

    def step(self, lr):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_size, out_size):
        self.W = np.random.randn(in_size, out_size)
        self.b = np.random.randn(out_size)
        self.dW = 0
        self.db = 0
        self.W_info = {}
        self.b_info = {}

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad):
        self.dW += np.dot(self.x.T, grad)
        self.db += np.sum(grad, axis=0)

        grad = np.dot(grad, self.W.T)
        return grad

    def zero_grad(self):
        self.dW = 0
        self.db = 0

    @property
    def parameter(self):
        return {
            "W": (self.W, self.dW, self.W_info),
            "b": (self.b, self.db, self.b_info),
        }

    @parameter.setter
    def parameter(self, parameter):
        self.W, self.dW, self.W_info = parameter["W"]
        self.b, self.db, self.b_info = parameter["b"]
