import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4, num_layers=2):
        super(TransformerModel, self).__init__()

        self.input_proj = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 39, 20, hidden_dim))
        self.dropout = nn.Dropout(0.2)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim * 39 * 20, num_classes)

    def forward(self, x):
        # x has shape [batch_size, 1, 39, 20]
        x = self.input_proj(x)  # shape [batch_size, hidden_dim, 39, 20]
        x = x.permute(0, 2, 3, 1)  # shape [batch_size, 39, 20, hidden_dim]
        # shape [batch_size, 39, 20, hidden_dim]
        x = x + self.positional_encoding
        # shape [batch_size, 39*20, hidden_dim]
        x = x.flatten(start_dim=1, end_dim=2)
        x = x.transpose(0, 1)  # shape [39*20, batch_size, hidden_dim]
        x = self.dropout(x)
        # shape [39*20, batch_size, hidden_dim]
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # shape [batch_size, 39*20, hidden_dim]
        x = x.flatten(start_dim=1)  # shape [batch_size, 39*20*hidden_dim]
        x = self.fc(x)  # shape [batch_size, num_classes]

        return x
