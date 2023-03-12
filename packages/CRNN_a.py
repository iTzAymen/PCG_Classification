import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(64)
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(RNN, self).__init__()
        self.maxpool1 = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size // 2, hidden_size, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # [batch_size, 1, n_mfcc, seq_length]
        out = x.squeeze(dim=1)
        # [batch_size, n_mfcc, seq_length]
        out = out.permute(0, 2, 1)
        # [batch_size, seq_length, n_mfcc]
        out = self.maxpool1(out)
        out, _ = self.lstm(out)
        out = self.flatten(out)
        return out


class CRNN_a(nn.Module):
    def __init__(self, input_dim, input_size, output_dim):
        super(CRNN_a, self).__init__()
        self.cnn = CNN(input_dim)
        self.rnn = RNN(input_size, 64)
        self.fc1 = nn.Linear(1152 + 9984, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        cnn_out = self.cnn(x)
        rnn_out = self.rnn(x)
        out = torch.cat([cnn_out, rnn_out], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
