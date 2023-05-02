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
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        y, _ = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, (6, 1)))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=1, device="cuda:0"):
        super(RNN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = TimeDistributed(
            nn.LSTM(input_size, hidden_size, n_layers, batch_first=True), True
        )

    def forward(self, x):
        # [batch_size, features, 1, seq_length]
        out = x.squeeze(dim=2)
        # [batch_size, features, seq_length]
        out = out.permute(0, 2, 1)
        # [batch_size, seq_length, features]
        out = self.lstm(out)
        return out[:, -1, :]


class SCRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        input_size,
        output_dim,
        device="cuda:0",
        n_layers_rnn=64,
    ):
        super(SCRNN, self).__init__()
        self.cnn = CNN(input_dim)
        self.rnn = RNN(input_size, 64, n_layers_rnn, device=device)
        self.fc1 = nn.Linear(64, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        cnn_out = self.cnn(x)  # [64, 64, 1, 8]
        rnn_out = self.rnn(cnn_out)  # [64, 64]
        out = self.fc1(rnn_out)  # [64, 32]
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)  # [64, 2]
        return out


# cnn = CNN(1)
# rnn = RNN(64, n_layers=64, device="cpu")
# model = SCRNN(1, 64, 2, device="cpu")

# tensor_cnn = torch.rand([64, 1, 64, 157])
# result = model(tensor_cnn)

# print(result.shape)
