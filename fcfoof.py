import torch
import torchvision


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=5, out_features=3)
        self.lin2 = torch.nn.Linear(in_features=3, out_features=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.nn.functional.relu(x)
        x = self.lin2(x)
        return x

    def get_param_layer(self):
        return [self, self.lin1, self.lin2]


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
        for layer in param_layers[1:]:
            in_feat = layer.in_features
            layer.register_buffer('input',
                                  torch.zeros(in_feat, in_feat, device=torch.device("cuda")))
            layer.register_buffer('exp', torch.zeros(in_feat, in_feat, device=torch.device("cuda")))
            layer.register_buffer('p', torch.eye(in_feat, device=torch.device("cuda")))
            layer.register_buffer('param',
                                  torch.zeros(in_feat, in_feat, device=torch.device("cuda")))

            def forward_hook(model, inp, outp):
                model.input = torch.matmul(inp[0].T, inp[0])

            def back_hook(model, gin, gout):
                _, _, g_param = gin
                model.param = torch.matmul(model.p, g_param)
                if self.t % self.defaults['inverse'] == 0:
                    model.p = torch.linalg.inv(
                        model.exp + self.defaults['damp'] * torch.eye(model.in_features, device=torch.device("cuda")))
                if (self.t + self.defaults['update']) % self.defaults['update'] < self.defaults['update']:
                    model.exp = self.defaults['decay'] * model.exp + (1 - self.defaults['decay']) * model.input
                self.t += 1

            layer.register_forward_hook(forward_hook)
            layer.register_backward_hook(back_hook)

    def step(self, closure=None):
        if closure is not None:
            closure()
        for layer in self.layer:
            for parameter in layer.parameters():
                if parameter.shape == layer.param.shape:
                    parameter.data -= self.defaults['lr'] * layer.param * parameter.grad.data
                else:
                    parameter.data -= self.defaults['lr'] * parameter.grad.data


if __name__ == '__main__':
    model = Model()
    optim = FOOF(model.get_param_layer(), 0.01, 100, 0.9, 5, 3)
    loss = torch.nn.MSELoss()

    inputs = torch.Tensor([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [3, 4, 5, 6, 1],
        [4, 5, 6, 2, 3]
    ])
    outputs = torch.Tensor([
        [1],
        [2],
        [1.5],
        [2.5]
    ])
    model = model.cuda()
    for i in range(20):
        print(i)
        inputs = inputs.cuda()
        outputs = outputs.cuda()
        pred = model(inputs)
        diff = loss(pred, outputs)
        optim.zero_grad()
        diff.backward()
        optim.step()
