import torch
import torch.nn as nn
from torch.nn import init


class AudioClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        conv_layers = []

        # first conv block
        self.conv1 = nn.Conv2d(input_dim, 8, kernel_size=(
            5, 5), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # second conv block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # third conv block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # fourth conv block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # drop out of 0.5
        self.dropout = nn.Dropout2d(0.5)
        # linear classifier
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(in_features=64, out_features=32)
        self.relu5 = nn.ReLU()
        self.lin2 = nn.Linear(in_features=32, out_features=16)
        self.relu6 = nn.ReLU()
        self.lin3 = nn.Linear(in_features=16, out_features=output_dim)
        self.relu7 = nn.ReLU()

        # wrap the convolutional blocks
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)
        x = self.dropout(x)
        # flatten for input to linear layer
        x = self.flatten(x)

        # Linear layer
        x = self.lin1(x)
        x = self.relu5(x)
        x = self.lin2(x)
        x = self.relu6(x)
        x = self.lin3(x)
        # Final output
        return x
