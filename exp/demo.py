import nn
import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Network):
    def __init__(self):
        super().__init__()
        self.add(nn.Linear(1, 16))
        self.add(nn.ReLU())
        self.add(nn.Linear(16, 16))
        self.add(nn.ReLU())
        self.add(nn.Linear(16, 1))


if __name__ == "__main__":
    np.random.seed(0)

    network = Network()
    criterion = nn.MSE(network.backward)
    # optimizer = nn.SGD(network.layers, lr=0.01, weight_decay=0.0, momentum=0.001, nesterov=True)
    optimizer = nn.Adam(network.layers, lr=0.01)

    x = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = x**3

    for epoch in range(20000):
        optimizer.zero_grad()
        pred = network(x)
        loss = criterion(pred, y)
        criterion.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")

    pred = network(x)
    plt.plot(x, y, label="true")
    plt.plot(x, pred, label="pred")
    plt.legend()
    plt.savefig("result.png")
