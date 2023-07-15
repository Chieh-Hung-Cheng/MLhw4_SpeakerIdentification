import torch
import torch.nn as nn
from torch.optim import Optimizer
import math
from torch.optim.lr_scheduler import LambdaLR


class SpeakerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(40, 80),
            # Medium Baseline: Change to Conformer
            ConformerBlock(80)
        )
        self.fc = nn.Sequential(
            nn.Linear(80, 80),
            nn.ReLU(),
            nn.Linear(80, 600)
        )

    def forward(self, x):
        # (Batch_size, N_len, 40)
        x = self.net(x)
        # (B, N, 80)
        x = x.mean(dim=1)
        # (B, 80) original dimension 1 collapsed
        x = self.fc(x)
        # (B, 600)
        return x


class MHSA_Module(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        self.query_matrix = nn.Linear(dim, dim)
        self.key_matrix = nn.Linear(dim, dim)
        self.value_matrix = nn.Linear(dim, dim)

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.layernorm(x)
        query = self.query_matrix(x)  # (B, N, dim)
        key = self.key_matrix(x)  # (B, N, dim)
        value = self.value_matrix(x)  # (B, N, dim)
        x, x_weights = self.attention(query, key, value)
        x = self.dropout(x)
        return x


class ConvolutionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layernorm = nn.LayerNorm(dim)
        self.conv_pointwise = nn.Conv1d(dim, dim, kernel_size=1) # bug here
        self.glu = nn.GLU()
        self.conv_depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=80)
        self.bn = nn.BatchNorm1d(dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        # input: (B, N, dim)
        x = self.layernorm(x)
        # (B, N, dim)
        x = self.conv_pointwise(x)
        x = self.glu(x)
        x = self.conv_depthwise(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.conv_pointwise(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_layer = nn.Linear(dim, dim)
        self.MHSA_M = MHSA_Module(dim=dim)
        self.conv_module = ConvolutionModule(dim=dim)
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):
        # input: (B, N, 80)
        x = x + self.linear_layer(x) * 0.5
        # (B, N, 80)
        x = x + self.MHSA_M(x)
        # (B, N, 80)
        x = x + self.conv_module(x)
        # (B, N, 80)
        x = x + self.linear_layer(x) * 0.5
        # (B, N, 80)
        x = self.layer_norm(x)
        return x


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == "__main__":
    sc = SpeakerClassifier()
    x = torch.rand((16, 300, 40))
    print(sc(x).shape)
