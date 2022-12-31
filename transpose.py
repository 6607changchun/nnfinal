import torch

if __name__ == '__main__':
    test = torch.randn(3, 4, 5)
    print(test)
    test_t = torch.transpose(test, 1, 2)
    print(test_t)
