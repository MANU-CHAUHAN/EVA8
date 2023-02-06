import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchsummary import summary

cv = partial(nn.Conv2d, bias=False)
bn = nn.BatchNorm2d
relu = nn.ReLU


class S6_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            cv(3, 16, 3, padding=1),
            relu(),
            bn(16),
            cv(16, 16, 1),
            relu(),
            bn(16),
            cv(16, 32, 3, dilation=1, groups=1, padding=1),
            relu(),
            bn(32),
            cv(32, 32, 1),
            relu(),
            bn(32),
            nn.Dropout2d(0.05)
        )

        self.block2 = nn.Sequential(
            cv(32, 64, 3, padding=1, dilation=1, groups=32),
            relu(),
            bn(64),
            cv(64, 32, 1),
            relu(),
            bn(32),
            cv(32, 64, 3, dilation=1, padding=1, groups=16),
            relu(),
            bn(64),
            cv(64, 48, 1),
            relu(),
            bn(48),
            cv(48, 64, 3, padding=1, dilation=2, groups=8),
            relu(),
            bn(64),
            cv(64, 32, 1),
            relu(),
            bn(32),
            nn.Dropout2d(0.08)
        )

        self.block3 = nn.Sequential(
            cv(32, 64, 3, dilation=1, padding=1, groups=32, stride=1),
            relu(),
            bn(64),
            cv(64, 128, 3, groups=16, dilation=1, padding=1),
            relu(),
            bn(128),
            cv(128, 64, 1),
            relu(),
            bn(64),
            cv(64, 96, 3, padding=1, groups=32),
            relu(),
            bn(96),
            cv(96, 64, 1),
            relu(),
            bn(64),
            cv(64, 64, 3, padding=2, dilation=2, stride=2),
            relu(),
            bn(64),
            cv(64, 64, 1),
            relu(),
            bn(64),
            nn.Dropout2d(0.09)
        )

        self.block4 = nn.Sequential(
            cv(64, 96, 3, padding=0, groups=32, stride=1, dilation=2),
            bn(96),
            relu(),
            cv(96, 64, 1),
            bn(64),
            relu(),
            nn.Dropout2d(0.05),
            cv(64, 64, 3, groups=64, padding=0, dilation=2),  # depthwise (a)
            cv(64, 32, 1),  # pointwise for preceding depthwise (b)
            bn(32),
            relu(),
            cv(32, 48, 3, dilation=2, groups=8),
            relu(),
            bn(48),
            cv(48, 10, 1, stride=1),
            relu(),
            bn(10),
            nn.AdaptiveAvgPool2d(1)
            # cv(10, 10, 1)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def test():
    model = S6_CIFAR10()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.size())
    print(summary(model, (3, 32, 32)))


if __name__ == '__main__':
    test()
