"""Small ResNet1D implementation for ECG signals.

This implementation is intentionally compact for experiments and educational use.
"""
import torch
import torch.nn as nn


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNReLU(channels, channels, kernel_size=3),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class ResNet1D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, channels=(16, 32, 64), dropout_p=0.0):
        super().__init__()
        layers = []
        prev_ch = in_channels
        for c in channels:
            layers.append(_ConvBNReLU(prev_ch, c, kernel_size=7, padding=3))
            layers.append(ResBlock(c))
            layers.append(nn.MaxPool1d(2))
            prev_ch = c

        self.features = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_ch, 64),
            nn.ReLU(inplace=True),
            self.dropout,
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, C, L)
        h = self.features(x)
        h = self.global_pool(h)
        return self.classifier(h)


def build_resnet1d(in_channels=1, num_classes=1, dropout_p=0.0):
    return ResNet1D(in_channels, num_classes, dropout_p=dropout_p)
