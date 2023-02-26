import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, kernel_size=3)
        self.mp1 = nn.MaxPool2d((2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.mp2 = nn.MaxPool2d((4, 4))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, output_dim, kernel_size=3)
        self.mp3 = nn.MaxPool2d((2, 2))
        self.bn3 = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        y = self.conv1(x)
        y = self.mp1(y)
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.mp2(y)
        y = self.bn2(y)
        y = self.conv3(y)
        y = self.mp3(y)
        y = self.bn3(y)
        return y


class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=64):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_dim * num_layers, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class CRNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CRNN, self).__init__()
        self.cnn = CNN(input_dim, 64)
        self.rnn = RNN(8, 32, output_dim, 64)

    def forward(self, x):
        out = self.cnn(x)
        out = out[:, :, -1, :]
        out = self.rnn(out)

        return out


class CRNN2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CRNN, self).__init__()
        self.cnn_spec = CNN(input_dim, 64)
        self.rnn_spec = RNN(8, 32, output_dim, 64)

        self.cnn_mfcc = CNN(input_dim, 64)
        self.rnn_mfcc = RNN(8, 32, output_dim, 64)

        self.lin = nn.Sequential(
            nn.Linear(4, 16),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        spec, mfcc = x

        spec_out = self.cnn_spec(spec)
        spec_out = spec_out[:, :, -1, :]
        spec_out = self.rnn_spec(spec_out)

        mfcc_out = self.cnn_mfcc(mfcc)
        mfcc_out = mfcc_out[:, :, -1, :]
        mfcc_out = self.rnn_mfcc(mfcc_out)
        temp = torch.cat([spec_out, mfcc_out], dim=1)
        out = self.lin(temp)
        return out
