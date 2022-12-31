# args
# kernel : (c_in, c_out, kernel...)
# image : (batch, c_in, image...)
# im2col(unfold) : (batch, c_in*kernel..., *image...)
#
# target
# image : (batch*image..., c_in*kernel...)
# kernel : (c_in*kernel..., c_out)
#
# result
# raw : (batch*image..., c_out)
# reshape(final) : (batch, c_out, image...)
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets


def get_divisor(input_shape, kernel_size, padding, dilation, stride) -> torch.Tensor:
    return torch.nn.functional.fold(
        torch.nn.functional.unfold(
            torch.ones(input_shape),
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride
        ),
        output_size=(input_shape[-2], input_shape[-1]),
        kernel_size=kernel_size,
        padding=padding,
        dilation=dilation,
        stride=stride
    )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=28, padding=0, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        return x

    def get_param_layer(self):
        return [self, self.conv1, self.conv2]


class FOOF(torch.optim.Optimizer):
    def __init__(self, param_layers: list[torch.nn.Module], lr: float, damp: float, decay: float,
                 inverse: int,
                 update: int):
        if len(param_layers) == 0:
            raise RuntimeError
        if update > inverse:
            raise RuntimeError
        super().__init__(param_layers[0].parameters(), {
            'lr': lr,
            'damp': damp,
            'decay': decay,
            'inverse': inverse,
            'update': update
        })
        self.t = 0
        self.layer = param_layers[1:]
        for layer in self.layer:
            layer.register_buffer('input', torch.zeros(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                       layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                       device=torch.device("cuda")))
            layer.register_buffer('exp', torch.zeros(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                     layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                     device=torch.device("cuda")))
            layer.register_buffer('p', torch.eye(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                 device=torch.device("cuda")))
            layer.register_buffer('param', torch.zeros(layer.out_channels,
                                                       layer.in_channels,
                                                       layer.kernel_size[0],
                                                       layer.kernel_size[1],
                                                       device=torch.device("cuda")))

            def forward_hook(model, inp, outp):
                img = inp[0]
                # img2col
                img = torch.nn.functional.unfold(
                    img,
                    kernel_size=model.kernel_size,
                    padding=model.padding,
                    stride=model.stride,
                    dilation=model.dilation
                )
                img = torch.transpose(img, 1, 2)
                img = torch.reshape(img, shape=(img.shape[0] * img.shape[1], img.shape[2]))
                # install buffer
                model.input = torch.matmul(img.T, img)

            def backward_hook(model, gin, gout):
                grad_param = gin[1]
                # extract convolution kernel
                grad_param = torch.reshape(grad_param, shape=(
                    grad_param.shape[0],
                    grad_param.shape[1] * grad_param.shape[2] * grad_param.shape[3]
                ))
                grad_param = torch.transpose(grad_param, 0, 1)
                # real multiply
                grad_update = torch.matmul(model.exp, grad_param)
                # enfold to kernel shape
                grad_update = torch.transpose(grad_update, 0, 1)
                grad_update = torch.reshape(grad_update, shape=(
                    model.out_channels,
                    model.in_channels,
                    model.kernel_size[0],
                    model.kernel_size[1]
                ))
                model.param = grad_update
                # update internal state
                if self.t % self.defaults['inverse'] == 0:
                    model.p = torch.linalg.inv(
                        model.exp + self.defaults['damp'] * torch.eye(
                            model.in_channels * model.kernel_size[0] * model.kernel_size[1],
                            device=torch.device("cuda")))
                if (self.t + self.defaults['update']) % self.defaults['update'] < self.defaults['update']:
                    model.exp = self.defaults['decay'] * model.exp + (1 - self.defaults['decay']) * model.input
                self.t += 1

            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(backward_hook)

    def step(self, closure=None):
        if closure is not None:
            closure()
        for layer in self.layer:
            for parameter in layer.parameters():
                if parameter.shape == layer.param.shape:
                    parameter.data -= self.defaults['lr'] * layer.param
                else:
                    parameter.data -= self.defaults['lr'] * parameter.grad.data


if __name__ == '__main__':
    device = "cuda"
    model = Model()
    loss_f = torch.nn.MSELoss()
    optim = FOOF(model.get_param_layer(),
                 lr=0.01,
                 damp=100,
                 decay=0.9,
                 inverse=5,
                 update=3
                 )

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

    model = model.to(device)
    for i in range(20):
        print(i)
        for x, label in train_loader:
            x = x.to(device)
            label = label.to(device).float()
            pred = model(x)
            loss = loss_f(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
