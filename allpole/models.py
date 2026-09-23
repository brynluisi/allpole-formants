"""Networks for SMELP (CNN encoder -> LARs -> LSTM decoder) and the LP-LSTM baseline.

Attribute names match the original training code so existing checkpoints load.
"""
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5)):
        super().__init__()
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class LarEncoder(nn.Module):
    """2-D CNN: log-magnitude STFT (B, 257, T) -> log-area ratios (B, lpc_order, T)."""

    def __init__(self, lpc_order=16, n_freqs=257, base_channels=16, max_channels=128, num_layers=5):
        super().__init__()
        channels = [min(base_channels * 2 ** i, max_channels) for i in range(num_layers + 1)]
        self.encoder = nn.Sequential(*[ConvBlock2D(1 if i == 0 else channels[i], channels[i + 1])
                                       for i in range(num_layers)])
        self.channel_reducer = nn.Conv2d(channels[num_layers], 1, kernel_size=1)
        self.final_layer = nn.Conv1d(n_freqs, lpc_order, kernel_size=1)

    def forward(self, spec):
        h = self.channel_reducer(self.encoder(spec.unsqueeze(1))).squeeze(1)
        return self.final_layer(h)


class FormantDecoder(nn.Module):
    """Bidirectional LSTM: per-frame features (T, B, input_size) -> log formants (T, B, 4)."""

    def __init__(self, input_size=16, hidden_size=128, num_layers=2, out_features=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, out_features)

    def forward(self, x):
        return self.fc(self.lstm(x)[0])
