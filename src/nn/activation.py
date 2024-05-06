import numpy as np


class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)
