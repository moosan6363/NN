import nn
import numpy as np
import matplotlib.pyplot as plt


class Network(nn.Network):
    def __init__(self):
        super().__init__()
        self.add(nn.Linear(2, 16))
        self.add(nn.ReLU())
        self.add(nn.Linear(16, 16))
        self.add(nn.ReLU())
        self.add(nn.Linear(16, 1))


if __name__ == "__main__":
    np.random.seed(0)

    network = Network()
    criterion = nn.MSE(network.backward)
    # optimizer = nn.SGD(
    #     network.layers, lr=0.01, weight_decay=0.0, momentum=0.1, nesterov=True
    # )
    optimizer = nn.Adam(network.layers, lr=0.01)

    x = np.random.rand(1000, 2)
    z = (1 + np.sin(4 * np.pi * x[:, 0])) * x[:, 1] / 2
    z = np.expand_dims(z, axis=1)

    for epoch in range(20000):
        optimizer.zero_grad()
        pred = network(x)
        loss = criterion(pred, z)
        criterion.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")

    z_pred = network(x)
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})
    ax.scatter(x[:, 0], x[:, 1], z, label="ground_truth", color="red")
    ax.scatter(x[:, 0], x[:, 1], z_pred, label="prediction", color="blue")
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.set_zlabel("z")
    fig.savefig("result.png")
