from torch import nn
import torch
import os


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        conv_layers = []

        # 3*448*448 -> 192*112*112
        leaky_relu = nn.LeakyReLU(0.1)
        conv_layers.append(nn.Conv2d(3, 192, 7, 2, padding=3))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.MaxPool2d(2))

        # 192*112*112 -> 256*56*56
        conv_layers.append(nn.Conv2d(192, 256, 3, 1, padding=1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.MaxPool2d(2))

        # 256*56*56 -> 512*28*28
        conv_layers.append(nn.Conv2d(256, 128, 1, 1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(128, 256, 3, 1, padding=1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(256, 256, 1, 1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(256, 512, 3, 1, padding=1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.MaxPool2d(2))

        # 512*28*28 -> 1024*14*14
        for _ in range(4):
            conv_layers.append(nn.Conv2d(512, 256, 1, 1))
            conv_layers.append(leaky_relu)
            conv_layers.append(nn.Conv2d(256, 512, 3, 1, padding=1))
            conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(512, 1024, 1, 1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(1024, 1024, 3, 1, padding=1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.MaxPool2d(2))

        # 1024*14*14 -> 1024*7*7
        for _ in range(2):
            conv_layers.append(nn.Conv2d(1024, 512, 1, 1))
            conv_layers.append(leaky_relu)
            conv_layers.append(nn.Conv2d(512, 1024, 3, 1, padding=1))
            conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(1024, 1024, 3, 1, padding=1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(1024, 1024, 3, 2, padding=1))
        conv_layers.append(leaky_relu)

        # 1024*7*7 -> 1024*7*7
        conv_layers.append(nn.Conv2d(1024, 1024, 3, 1, padding=1))
        conv_layers.append(leaky_relu)
        conv_layers.append(nn.Conv2d(1024, 1024, 3, 1, padding=1))
        conv_layers.append(leaky_relu)

        # # 1024*7*7 -> 1*4096
        # conv_layers.append(nn.Flatten())
        # conv_layers.append(nn.Linear(7*7*1024, 4096))
        # conv_layers.append(leaky_relu)

        # # 1*4096 -> 1*1470(7*7*30)
        # conv_layers.append(nn.Linear(4096, 1470))
        # conv_layers.append(leaky_relu)

        # 30*7*7
        conv_layers.append(nn.Conv2d(1024, 30, 1, 1))

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        pred = self.conv_layers(x)
        pred = pred.view(-1, 7, 7, 30)
        return pred


if __name__ == '__main__':
    print('test net ...')
    x = torch.randn((1, 3, 448, 448))
    print(f'x shape: {x.shape}')

    model = Net()
    predictions = model(x)
    print(f'predictions shape: {predictions.shape}')

    os.system('pause')
