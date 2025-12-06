import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))

class EmotionMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, depth=4, dropout=0.3):
        super().__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ]
        for _ in range(depth):
            layers.append(ResidualBlock(hidden_dim, dropout=dropout))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)