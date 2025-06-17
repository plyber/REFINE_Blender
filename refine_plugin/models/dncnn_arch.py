import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        # first layer
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        # middle layers
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # last layer
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise  # residual
