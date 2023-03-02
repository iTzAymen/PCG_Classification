import torch
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        # (samples * timesteps, input_size)
        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class RNN(nn.Module):
    # shape of data for RNN is (batch_size, time, mfcc_n)
    def __init__(self, input_dim, output_dim, num_layers=64, sequence_length=157):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128,
                            num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128,
                             num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.td1 = TimeDistributed(nn.Linear(128, 64))
        self.relu1 = nn.ReLU()
        self.td2 = TimeDistributed(nn.Linear(64, 32))
        self.relu2 = nn.ReLU()
        self.td3 = TimeDistributed(nn.Linear(32, 16))
        self.relu3 = nn.ReLU()
        self.td4 = TimeDistributed(nn.Linear(16, 8))
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(sequence_length * 8, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 2)

    def forward(self, x):
        # shape is (batch_size, mfcc_n, time)
        out = x.permute(0, 2, 1)
        # shape is (batch_size, time, mfcc_n)
        out, _ = self.lstm(out)
        out, _ = self.lstm2(out)
        out = self.dropout(out)
        out = self.td1(out)
        out = self.relu1(out)
        out = self.td2(out)
        out = self.relu2(out)
        out = self.td3(out)
        out = self.relu3(out)
        out = self.td4(out)
        out = self.relu4(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.linear3(out)
        out = self.linear4(out)
        return out


# model = RNN(13 * 3, 2, sequence_length=157)
