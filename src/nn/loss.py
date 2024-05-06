import numpy as np


class MSE:
    def __init__(self, backward_func):
        self.backward_func = backward_func

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        self.x = x
        self.y = y
        return np.mean((x - y) ** 2)

    def backward(self):
        grad = 2 * (self.x - self.y) / len(self.x)
        return self.backward_func(grad)
