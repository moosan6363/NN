import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


class Network(nn.Network):
    def __init__(self):
        super().__init__()
        self.add(nn.Linear(2, 16))
        self.add(nn.ReLU())
        self.add(nn.Linear(16, 16))
        self.add(nn.ReLU())
        self.add(nn.Linear(16, 1))


def train(x, z, epoch, batch_size):
    network = Network()
    criterion = nn.MSE(network.backward)
    optimizer = nn.Adam(network.layers, lr=0.01)

    for epoch in range(epoch):
        kf = KFold(n_splits=x.shape[0] // batch_size, shuffle=True, random_state=0)

        for train_index, _ in kf.split(x):
            x_batch, z_batch = x[train_index], z[train_index]
            optimizer.zero_grad()
            pred = network(x_batch)
            loss = criterion(pred, z_batch)
            criterion.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")
    return network


if __name__ == "__main__":
    np.random.seed(0)

    x = np.random.rand(32 * 5 * 10, 2)
    z = (1 + np.sin(4 * np.pi * x[:, 0])) * x[:, 1] / 2
    z = np.expand_dims(z, axis=1)

    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    loss_list = []

    for train_index, test_index in cv.split(x):
        x_train, x_test = x[train_index], x[test_index]
        z_train, z_test = z[train_index], z[test_index]
        network = train(x_train, z_train, 2000, 32)
        z_pred = network(x_test)
        loss = nn.MSE(network.backward)(z_pred, z_test)
        loss_list.append(loss)

        print(f"loss: {loss}")

    print(f"mean loss: {np.mean(loss_list)}, std loss: {np.std(loss_list)}")
