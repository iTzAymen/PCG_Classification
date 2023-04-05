import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes, device='cpu'):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size,
                            n_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 32)
        self.linear2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, n_classes)

    def forward(self, x):
        # [batch_size, 1, n_mels, seq_length]
        out = x.squeeze(dim=1)
        # [batch_size, n_mels, seq_length]
        out = out.permute(0, 2, 1)

        hidden_states = torch.zeros(self.n_layers, out.size(
            0), self.hidden_size).to(self.device)
        cell_states = torch.zeros(self.n_layers, out.size(
            0), self.hidden_size).to(self.device)

        # [batch_size, seq_length, n_mels]
        out, _ = self.lstm(out, (hidden_states, cell_states))
        out = self.linear1(out[:, -1, :])
        out = self.linear2(out)
        out = self.output_layer(out)

        return out


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes, device='cpu'):
        super(BiLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size,
                            n_layers, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(hidden_size * 2, 32)
        self.linear2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, n_classes)

    def forward(self, x):
        # [batch_size, 1, n_mels, seq_length]
        out = x.squeeze(dim=1)
        # [batch_size, n_mels, seq_length]
        out = out.permute(0, 2, 1)

        hidden_states = torch.zeros(self.n_layers * 2, out.size(
            0), self.hidden_size).to(self.device)
        cell_states = torch.zeros(self.n_layers * 2, out.size(
            0), self.hidden_size).to(self.device)

        # [batch_size, seq_length, n_mels]
        out, _ = self.lstm(out, (hidden_states, cell_states))
        out = self.linear1(out[:, -1, :])
        out = self.linear2(out)
        out = self.output_layer(out)

        return out


# tensor = torch.rand([128, 1, 128, 157])
# model = BiLSTM(input_size=128, hidden_size=64, n_layers=32, n_classes=2)


# result = model(tensor)
# print(result.shape)
