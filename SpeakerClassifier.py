import torch
import torch.nn as nn


class SpeakerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 80),
            # Medium Baseline: Change to Conformer
            nn.TransformerEncoderLayer(d_model=80, nhead=1, dim_feedforward=256, batch_first=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 600)
        )

    def forward(self, x):
        # Input x is (B,mel_len,40)
        x = self.net(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x