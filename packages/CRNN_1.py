import torch
import torch.nn as nn

# Define the CNN layer
class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.conv(x)
        out = self.fc(out)
        return out

# Define the RNN layer
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, bidirectional=True):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[-1, :, :])
        return out

# Define the overall model
class PCGClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim_cnn, output_dim_cnn, hidden_dim_rnn, num_layers, output_dim):
        super(PCGClassifier, self).__init__()
        self.cnn = CNN(input_dim, hidden_dim_cnn, output_dim_cnn)
        self.rnn = RNN(output_dim_cnn, hidden_dim_rnn, num_layers, output_dim)

    def forward(self, x):
        out = self.cnn(x)
        out = out.unsqueeze(0)
        out = self.rnn(out)
        return out
