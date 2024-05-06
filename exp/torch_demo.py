import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    net = Net(1, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    x = torch.linspace(-1, 1, 100).view(-1, 1)
    y = x**3

    for epoch in range(1000):
        optimizer.zero_grad()
        pred = net(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss.item()}")
    pred = net(x)

    plt.plot(x.numpy(), y.numpy(), label="true")
    plt.plot(x.numpy(), pred.detach().numpy(), label="pred")
    plt.legend()
    plt.savefig("result.png")
