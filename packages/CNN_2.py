import torch
import torch.nn as nn
from torch.nn import init


class AudioClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        conv_layers = []

        # first conv block
        self.conv1 = nn.Conv2d(input_dim, 16, kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.mp1 = nn.MaxPool2d(2)

        conv_layers += [self.conv1, self.relu1, self.mp1]

        # second conv block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        self.mp2 = nn.MaxPool2d(2)

        conv_layers += [self.conv2, self.relu2, self.mp2]

        # third conv block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        self.mp3 = nn.MaxPool2d(2)

        conv_layers += [self.conv1, self.relu1, self.mp1]

        # global maxpool
        self.gmp = nn.MaxPool2d(2)

        # linear classifier
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.lin2 = nn.Linear(32, out_features=output_dim)

        # wrap the convolutional blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # global maxpool and flatten for input to linear layer
        x = self.gmp(x)
        x = self.flatten(x)

        # Linear layer
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.lin2(x)

        # Final output
        return x
