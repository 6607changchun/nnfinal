import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv1_p = torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2)
        self.conv2_p = torch.nn.Conv2d(in_channels=10, out_channels=1, kernel_size=1, padding=0)
        self.lin1 = torch.nn.Linear(in_features=28 * 28, out_features=84)
        self.lin2 = torch.nn.Linear(in_features=84, out_features=1)

    def forward(self, x):
        batch = x.size(0)
        output1 = torch.nn.functional.relu(self.conv1_p(torch.nn.functional.relu(self.conv1(x))))
        output2 = torch.nn.functional.relu(self.conv2_p(torch.nn.functional.relu(self.conv2(x))))
        output = (output2 + output1) / 2
        output = output.view(batch, -1)
        y = torch.nn.functional.relu(self.lin1(output))
        y = torch.nn.functional.relu(self.lin2(y))
        return y


def my_loss(x, y):
    return torch.sum(torch.pow(x - y, 2)) / 2


if __name__ == '__main__':
    device = "cuda"
    model = Model().to(device)
    optim = torch.optim.Adam(model.parameters())

    train_set = datasets.MNIST(root="data", train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]))
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]))

    train_loader = DataLoader(dataset=train_set, batch_size=3000, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_set, batch_size=3000, shuffle=True, num_workers=4)

    for i in range(10):
        print(i)
        for x, label in train_loader:
            x = x.to(device)
            label = label.to(device)
            output = model(x)
            loss = my_loss(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
    correct = 0
    with torch.no_grad():
        for x, label in test_loader:
            x = x.to(device)
            label = label.to(device)
            pred = model(x)
            correct += len([0 for i in range(len(label)) if torch.abs(label[i] - pred[i]) <= 0.5])

    print("{}%".format(correct * 100.0 / len(test_set)))
