import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(4), nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(64)
        )
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, (2, 1)))

        self.Initialize_weights()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out

    def Initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight)
                nn.init.constant_(m.bias, 0)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=1, device="cuda:0"):
        super(RNN, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # [batch_size, features, 1, seq_length]
        out = x.squeeze(dim=2)
        # [batch_size, features, seq_length]
        out = out.permute(0, 2, 1)
        hidden_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        cell_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        # [batch_size, seq_length, features]
        out, _ = self.lstm(out, (hidden_states, cell_states))
        out = self.flatten(out[:, -1, :])
        return out

    def Initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LSTM):
                nn.init.kaiming_uniform(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


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
