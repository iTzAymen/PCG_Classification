import torch
from torch import nn


# [batch_size, 1, 128, 157]
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, n_classes, device="cpu"):
        super(BiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size, hidden_size, n_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        out = x.squeeze(dim=1)
        out = out.permute(0, 2, 1)
        h0 = torch.zeros(self.n_layers * 2, out.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.n_layers * 2, out.size(0), self.hidden_size).to(
            self.device
        )

        out, _ = self.lstm(out, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = BiLSTM(128, 128, 64, 2, device=device).to(device)
# tensor = torch.rand([64, 1, 128, 157]).to(device)

# result = model(tensor)
# print(result.shape)
