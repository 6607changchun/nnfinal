# Convolution:
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
import time

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib
from matplotlib import pyplot

matplotlib.use('QtAgg')

device = 'cuda'


class FOOF(torch.optim.Optimizer):
    def __init__(self,
                 param_layers: list[torch.nn.Module],
                 lr: float,
                 damp: float,
                 decay: float,
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
            if isinstance(layer, torch.nn.Linear):
                in_feat = layer.in_features
                layer.register_buffer('input',
                                      torch.zeros(in_feat, in_feat, device=torch.device(device)))
                layer.register_buffer('exp', torch.zeros(in_feat, in_feat, device=torch.device(device)))
                layer.register_buffer('p', torch.eye(in_feat, device=torch.device(device)))
                layer.register_buffer('param',
                                      torch.zeros(in_feat, in_feat, device=torch.device(device)))

                def forward_hook_lin(model, inp, outp):
                    model.input = torch.matmul(inp[0].T, inp[0])

                def back_hook_lin(model, gin, gout):
                    _, _, g_param = gin
                    model.param = torch.matmul(model.p, g_param)
                    if self.t % self.defaults['inverse'] == 0:
                        # try:
                        model.p = torch.linalg.inv(
                            model.exp + self.defaults['damp'] * torch.eye(model.in_features,
                                                                          device=torch.device(device)))
                        #
                        # except Exception:
                        #     debug = model.exp + self.defaults['damp'] * torch.eye(model.in_features,
                        #                                                       device=torch.device(device))
                        #     print(debug)
                        #     assert False
                    if (self.t + self.defaults['update']) % self.defaults['update'] < self.defaults['update']:
                        model.exp = self.defaults['decay'] * model.exp + (1 - self.defaults['decay']) * model.input
                    self.t += 1

                layer.register_forward_hook(forward_hook_lin)
                layer.register_backward_hook(back_hook_lin)
            elif isinstance(layer, torch.nn.Conv2d):
                layer.register_buffer('input',
                                      torch.zeros(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                  layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                  device=torch.device(device)))
                layer.register_buffer('exp',
                                      torch.zeros(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                  layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                  device=torch.device(device)))
                layer.register_buffer('p', torch.eye(layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1],
                                                     device=torch.device(device)))
                layer.register_buffer('param', torch.zeros(layer.out_channels,
                                                           layer.in_channels,
                                                           layer.kernel_size[0],
                                                           layer.kernel_size[1],
                                                           device=torch.device(device)))

                def forward_hook_conv2d(model, inp, outp):
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

                def backward_hook_conv2d(model, gin, gout):
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
                                device=torch.device(device)))
                    if (self.t + self.defaults['update']) % self.defaults['update'] < self.defaults['update']:
                        model.exp = self.defaults['decay'] * model.exp + (1 - self.defaults['decay']) * model.input
                    self.t += 1

                layer.register_forward_hook(forward_hook_conv2d)
                layer.register_backward_hook(backward_hook_conv2d)
            else:
                raise NotImplementedError('other ops are not supported')

    def step(self, closure=None):
        if closure is not None:
            closure()
        for layer in self.layer:
            if isinstance(layer, torch.nn.Linear):
                for parameter in layer.parameters():
                    if parameter.shape == layer.param.shape:
                        parameter.data -= self.defaults['lr'] * layer.param * parameter.grad.data
                    else:
                        parameter.data -= self.defaults['lr'] * parameter.grad.data
            elif isinstance(layer, torch.nn.Conv2d):
                for parameter in layer.parameters():
                    if parameter.shape == layer.param.shape:
                        parameter.data -= self.defaults['lr'] * layer.param
                    else:
                        parameter.data -= self.defaults['lr'] * parameter.grad.data


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv1_d = torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, padding=2)
        self.conv2_d = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
        self.lin1 = torch.nn.Linear(in_features=28 * 28 * 3, out_features=7 * 7 * 3)
        self.lin2 = torch.nn.Linear(in_features=7 * 7 * 3, out_features=10)

    def forward(self, x):
        batch = x.size(0)
        in1 = self.conv1(x)
        in1 = torch.nn.functional.relu(in1)
        in1 = self.conv1_d(in1)
        in1 = torch.nn.functional.relu(in1)
        in2 = self.conv2(x)
        in2 = torch.nn.functional.relu(in2)
        in2 = self.conv2_d(in2)
        in2 = torch.nn.functional.relu(in2)
        in3 = self.conv3(x)
        in3 = torch.nn.functional.relu(in3)

        in1 = in1.view(batch, -1)
        in2 = in2.view(batch, -1)
        in3 = in3.view(batch, -1)
        in0 = torch.cat([in1, in2, in3], dim=1)

        pred = self.lin1(in0)
        pred = torch.nn.functional.relu(pred)
        pred = self.lin2(pred)
        pred = torch.nn.functional.relu(pred)
        pred = torch.nn.functional.log_softmax(pred, dim=1)
        return pred

    def get_param_layer(self):
        return [self, self.conv1, self.conv1_d, self.conv2, self.conv2_d, self.conv3, self.lin1, self.lin2]


def optim_mix(configs: list[tuple[str, torch.nn.Module, torch.nn.Module, torch.optim.Optimizer]], epoch: int,
              train_loader: DataLoader, test_loader: DataLoader, test_len, device='cuda'):
    for name, model, loss_f, optim in configs:
        print("model {}".format(name))
        start = time.perf_counter()
        model = model.to(device)
        loss_seq = []
        for i in range(epoch):
            print("epoch {}".format(i))
            loss_total = 0.0
            for x, label in train_loader:
                x = x.to(device)
                label = label.to(device)
                output = model(x)
                loss = loss_f(output, label)
                loss_total += loss.cpu().item()
                optim.zero_grad()
                loss.backward()
                optim.step()
            print("loss {}".format(loss_total / len(train_loader)))
            loss_seq.append(loss_total / len(train_loader))
            with torch.no_grad():
                correct = 0
                for x, label in test_loader:
                    x = x.to(device)
                    label = label.to(device)
                    pred = torch.argmax(model(x), dim=1)
                    correct += torch.sum(label == pred)
                print("acc {}%".format(correct * 100.0 / test_len))
        with torch.no_grad():
            correct = 0
            for x, label in test_loader:
                x = x.to(device)
                label = label.to(device)
                pred = torch.argmax(model(x), dim=1)
                correct += torch.sum(label == pred)
            print("final acc {}%".format(correct * 100.0 / test_len))
        end = time.perf_counter()
        print("total time {}".format(end - start))
        pyplot.plot(loss_seq, label="{}_loss".format(name))
        pyplot.legend()
    pyplot.show()


class ModelZoo:
    def __init__(self):
        self.model = {}

    def __getitem__(self, item):
        if item in self.model.keys():
            return self.model[item]
        else:
            model = Model()
            self.model[item] = model
            return model


if __name__ == '__main__':
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_set, batch_size=3000, shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset=test_set, batch_size=3000, shuffle=True, num_workers=4)

    model_zoo = ModelZoo()
    optim_mix(
        configs=[
            ('sgd/1e-3', model_zoo['sgd/1e-3'], torch.nn.CrossEntropyLoss(),
             torch.optim.SGD(model_zoo['sgd/1e-3'].parameters(), lr=1e-3)),
            ('sgd/1e-2', model_zoo['sgd/1e-2'], torch.nn.CrossEntropyLoss(),
             torch.optim.SGD(model_zoo['sgd/1e-3'].parameters(), lr=1e-2)),
            ('adam/default', model_zoo['adam/default'], torch.nn.CrossEntropyLoss(),
             torch.optim.Adam(model_zoo['adam/default'].parameters())),
            ('foof/0.01-100-0.9-5-3', model_zoo['foof/0.01-100-0.9-5-3'], torch.nn.CrossEntropyLoss(), FOOF(
                model_zoo['foof/0.01-100-0.9-5-3'].get_param_layer(),
                lr=0.01,
                damp=100,
                decay=0.9,
                inverse=5,
                update=3
            )),
            ('foof/0.01-10000-0.95-5-3', model_zoo['foof/0.01-10000-0.95-5-3'], torch.nn.CrossEntropyLoss(), FOOF(
                model_zoo['foof/0.01-10000-0.95-5-3'].get_param_layer(),
                lr=0.01,
                damp=10000,
                decay=0.95,
                inverse=5,
                update=3
            )),
            ('foof/0.01-100-0.9-5-5', model_zoo['foof/0.01-100-0.9-5-5'], torch.nn.CrossEntropyLoss(), FOOF(
                model_zoo['foof/0.01-100-0.9-5-5'].get_param_layer(),
                lr=0.01,
                damp=100,
                decay=0.9,
                inverse=5,
                update=5
            )),
            ('foof/0.01-100-0.99-5-3', model_zoo['foof/0.01-100-0.99-5-3'], torch.nn.CrossEntropyLoss(), FOOF(
                model_zoo['foof/0.01-100-0.99-5-3'].get_param_layer(),
                lr=0.01,
                damp=100,
                decay=0.99,
                inverse=5,
                update=3
            )),
            ('foof/0.01-100-0.99-1-1', model_zoo['foof/0.01-100-0.99-1-1'], torch.nn.CrossEntropyLoss(), FOOF(
                model_zoo['foof/0.01-100-0.99-1-1'].get_param_layer(),
                lr=0.01,
                damp=100,
                decay=0.99,
                inverse=1,
                update=1
            )),
        ],
        epoch=50,
        train_loader=train_loader,
        test_loader=test_loader,
        test_len=len(test_set)
    )
