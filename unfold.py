import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets

if __name__ == '__main__':
    device = "cuda"

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

    for x, label in train_loader:
        test_ones = torch.nn.functional.unfold(torch.ones(x.shape), kernel_size=3, padding=1, dilation=1, stride=1)
        divisor = torch.nn.functional.fold(test_ones, output_size=(x.shape[-2], x.shape[-1]), kernel_size=3, padding=1,
                                           dilation=1, stride=1)
        unfold = torch.nn.functional.unfold(x, kernel_size=3, padding=1, dilation=1, stride=1)
        fold = torch.nn.functional.fold(unfold, output_size=(28, 28), kernel_size=3, padding=1, dilation=1, stride=1)
        break

    test = torch.randn(2, 3, 4, 5)
    unfold = torch.nn.functional.unfold(test, kernel_size=3, padding=1)
    print(unfold.shape)
