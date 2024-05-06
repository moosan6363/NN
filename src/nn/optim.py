import numpy as np


class Optimizer:
    def __init__(self, layers):
        self.layers = layers

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        layers,
        lr=0.01,
        weight_decay=0.0,
        momentum=0.0,
        nesterov=False,
        dampening=0.0,
        maximize=False,
    ):
        super().__init__(layers)
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.dampening = dampening
        self.maximize = maximize

    def step(self):
        for layer in self.layers:
            if hasattr(layer, "parameter"):
                parameter = layer.parameter
                for key in parameter.keys():
                    theta, g, info = parameter[key]

                    if self.weight_decay != 0:
                        g += self.weight_decay * theta

                    if self.momentum != 0:
                        if "b" in info:
                            info["b"] = (
                                self.momentum * info["b"] + (1 - self.dampening) * g
                            )
                        else:
                            info["b"] = g

                        if self.nesterov:
                            g += self.momentum * info["b"]
                        else:
                            g = info["b"]

                    if self.maximize:
                        theta += self.lr * g
                    else:
                        theta -= self.lr * g


class Adam(Optimizer):
    def __init__(
        self,
        layers,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
    ):
        super().__init__(layers)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.t = 0

    def step(self):
        self.t += 1
        for layer in self.layers:
            if hasattr(layer, "parameter"):
                parameter = layer.parameter
                for key in parameter.keys():
                    theta, g, info = parameter[key]

                    if self.maximize:
                        g = -g

                    if self.weight_decay != 0:
                        g += self.weight_decay * theta

                    beta1, beta2 = self.betas

                    if "m" not in info:
                        info["m"] = 0
                        info["v"] = 0
                        if self.amsgrad:
                            info["v_hat_max"] = 0
                    else:
                        info["m"] = beta1 * info["m"] + (1 - beta1) * g
                        info["v"] = beta2 * info["v"] + (1 - beta2) * g**2

                    m_hat = info["m"] / (1 - beta1**self.t)
                    v_hat = info["v"] / (1 - beta2**self.t)

                    if self.amsgrad:
                        info["v_hat_max"] = np.maximum(info["v_hat_max"], v_hat)
                        theta -= (
                            self.lr * m_hat / (np.sqrt(info["v_hat_max"]) + self.eps)
                        )
                    else:
                        theta -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
