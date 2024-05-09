import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def training(
    train_data,
    valid_data,
    epoch,
    batch_size,
    NetworkClass,
    lr=0.01,
    weight_decay=0.0,
    validation_times=100,
):
    network = NetworkClass()
    criterion = nn.MSE(network.backward)
    optimizer = nn.Adam(network.layers, lr=lr, weight_decay=weight_decay)

    x_train, z_train = train_data
    x_valid, z_valid = valid_data

    epoch_hist = []
    train_loss_hist = []
    valid_loss_hist = []

    for i in range(epoch):
        kf = KFold(
            n_splits=x_train.shape[0] // batch_size, shuffle=True, random_state=0
        )

        for train_index, _ in kf.split(x_train):
            x_train_batch, z_train_batch = x_train[train_index], z_train[train_index]
            optimizer.zero_grad()
            pred = network(x_train_batch)
            loss = criterion(pred, z_train_batch)
            criterion.backward()
            optimizer.step()

        if i % validation_times == 0:
            epoch_hist.append(i)
            train_loss = criterion(network(x_train), z_train)
            valid_loss = criterion(network(x_valid), z_valid)

            train_loss_hist.append(train_loss)
            valid_loss_hist.append(valid_loss)

            print(f"epoch: {i}, train loss: {train_loss}, valid loss: {valid_loss}")

    return epoch_hist, train_loss_hist, valid_loss_hist


if __name__ == "__main__":
    np.random.seed(0)

    class Network(nn.Network):
        def __init__(self):
            super().__init__()
            self.add(nn.Linear(2, 16))
            self.add(nn.ReLU())
            self.add(nn.Linear(16, 1))

    train_loss_hists = []
    valid_loss_hists = []

    n_split = 5
    extrapolation = True

    if extrapolation:
        x_train = np.random.rand(32 * (n_split - 1) * 10, 2)
        z_train = (1 + np.sin(4 * np.pi * x_train[:, 0])) * x_train[:, 1] / 2
        z_train = np.expand_dims(z_train, axis=1)

        mean_shift = (1, 1)
        x_valid = np.random.rand(32 * 1 * 10, 2) + mean_shift
        z_valid = (1 + np.sin(4 * np.pi * x_valid[:, 0])) * x_valid[:, 1] / 2
        z_valid = np.expand_dims(z_valid, axis=1)

        epoch_hist, train_loss_hist, valid_loss_hist = training(
            (x_train, z_train),
            (x_valid, z_valid),
            200,
            32,
            Network,
            lr=0.01,
            weight_decay=0.01,
            validation_times=5,
        )
        train_loss_hists.append(train_loss_hist)
        valid_loss_hists.append(valid_loss_hist)
        print()
    else:
        x = np.random.rand(32 * n_split * 10, 2)
        z = (1 + np.sin(4 * np.pi * x[:, 0])) * x[:, 1] / 2
        z = np.expand_dims(z, axis=1)
        cv = KFold(n_splits=5, shuffle=True, random_state=0)

        for index, (train_index, valid_index) in enumerate(cv.split(x)):
            print(f"Cross Validation: {index + 1} / {n_split}")
            x_train, x_test = x[train_index], x[valid_index]
            z_train, z_test = z[train_index], z[valid_index]
            epoch_hist, train_loss_hist, valid_loss_hist = training(
                (x_train, z_train),
                (x_test, z_test),
                200,
                32,
                Network,
                lr=0.01,
                validation_times=5,
            )
            train_loss_hists.append(train_loss_hist)
            valid_loss_hists.append(valid_loss_hist)
            print()

        train_loss_hists = np.array(train_loss_hists)
        valid_loss_hists = np.array(valid_loss_hists)

        print(f"train_loss_mean: {np.mean(train_loss_hists, axis=0)[-1]}")
        print(f"valid_loss_mean: {np.mean(valid_loss_hists, axis=0)[-1]}")

    fig, ax = plt.subplots(1, 1, squeeze=False)
    ax[0, 0].plot(
        epoch_hist, np.mean(train_loss_hists, axis=0), label="train loss", color="red"
    )
    ax[0, 0].fill_between(
        epoch_hist,
        np.mean(train_loss_hists, axis=0) + np.std(train_loss_hists, axis=0),
        np.mean(train_loss_hists, axis=0) - np.std(train_loss_hists, axis=0),
        alpha=0.3,
    )

    ax[0, 0].plot(
        epoch_hist, np.mean(valid_loss_hists, axis=0), label="valid loss", color="blue"
    )
    ax[0, 0].fill_between(
        epoch_hist,
        np.mean(valid_loss_hists, axis=0) + np.std(valid_loss_hists, axis=0),
        np.mean(valid_loss_hists, axis=0) - np.std(valid_loss_hists, axis=0),
        alpha=0.3,
    )
    ax[0, 0].set_xlabel("epoch")
    ax[0, 0].set_ylabel("loss")
    ax[0, 0].legend()
    fig.savefig("result.png")
