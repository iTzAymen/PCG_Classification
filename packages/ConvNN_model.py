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


class ConvNN_model(nn.Module):
    def __init__(self):
        super(ConvNN_model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.lstm1 = TimeDistributed(nn.LSTM(75, 128), True)
        self.drop = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 128)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.lstm1(out)
        out = self.drop(out)
        out, _ = self.lstm2(out)
        out = self.layers(out[:, -1, :])
        return out


# model = ConvNN_model()
# tensor_cnn = torch.rand([64, 1, 64, 157])
# result = model(tensor_cnn)

# print(result.shape)
