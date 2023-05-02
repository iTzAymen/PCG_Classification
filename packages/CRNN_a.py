import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(16)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2), nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(4), nn.BatchNorm2d(64)
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


class CRNN_a_spec(nn.Module):
    def __init__(self, input_dim, input_size, output_dim, fc_in=6528):
        super(CRNN_a_spec, self).__init__()
        self.cnn = CNN(input_dim)
        self.rnn = RNN(input_size, 64)
        self.fc1 = nn.Linear(fc_in, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        cnn_out = self.cnn(x)
        print(cnn_out.shape)
        rnn_out = self.rnn(x)
        print(rnn_out.shape)
        out = torch.cat([cnn_out, rnn_out], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class RNN_b(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=1, device="cuda:0"):
        super(RNN_b, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.maxpool1 = nn.MaxPool2d(2)
        self.lstm = nn.LSTM(input_size // 2, hidden_size, n_layers, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # [batch_size, 1, n_mfcc, seq_length]
        out = x.squeeze(dim=1)
        # [batch_size, n_mfcc, seq_length]
        out = out.permute(0, 2, 1)
        hidden_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        cell_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        # [batch_size, seq_length, n_mfcc]
        out = self.maxpool1(out)
        out, _ = self.lstm(out, (hidden_states, cell_states))
        out = self.flatten(out)
        return out


class CRNN_b_spec(nn.Module):
    def __init__(
        self,
        input_dim,
        input_size,
        output_dim,
        fc_in=8576,
        device="cuda:0",
        n_layers_rnn=64,
    ):
        super(CRNN_b_spec, self).__init__()
        self.cnn = CNN(input_dim)
        self.rnn = RNN_b(input_size, 64, n_layers_rnn, device=device)
        self.fc1 = nn.Linear(fc_in, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
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


class RNN_c(nn.Module):
    def __init__(self, input_size, hidden_size=64, n_layers=1, device="cuda:0"):
        super(RNN_c, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.flatten = nn.Flatten()

    def forward(self, x):
        # [batch_size, 1, n_mfcc, seq_length]
        out = x.squeeze(dim=1)
        # [batch_size, n_mfcc, seq_length]
        out = out.permute(0, 2, 1)
        hidden_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        cell_states = torch.zeros(self.n_layers, out.size(0), self.hidden_size).to(
            self.device
        )
        # [batch_size, seq_length, n_mfcc]
        out, _ = self.lstm(out, (hidden_states, cell_states))
        out = self.flatten(out[:, -1, :])
        return out


class CRNN_c_spec(nn.Module):
    def __init__(
        self,
        input_size,
        n_classes,
        n_layers_rnn=64,
        fc_in=8576,
        device="cuda:0",
    ):
        super(CRNN_c_spec, self).__init__()
        self.cnn = CNN()
        self.rnn = RNN_c(input_size, 64, n_layers_rnn, device=device)
        self.fc1 = nn.Linear(fc_in, 32)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        cnn_out = self.cnn(x)
        rnn_out = self.rnn(x)
        out = torch.cat([cnn_out, rnn_out], dim=1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


# model = CRNN_c_spec(
#     input_size=128, n_classes=2, n_layers_rnn=64, fc_in=3648, device="cpu"
# )
# tensor = torch.rand([64, 1, 128, 157])

# print(model(tensor).shape)
