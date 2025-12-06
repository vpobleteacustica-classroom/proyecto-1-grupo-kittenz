import torch
import torch.nn as nn

class TinyMelCNNMinimal(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        # Arquitectura mínima: 2 convs pequeñas y un head global
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # reduce H,T a la mitad
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # vuelve a reducir
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),  # Global average
            nn.Flatten(),
            nn.Linear(64, num_classes)    # sin dropout, sin BN
        )

    def forward(self, x):
        z = self.features(x)
        logits = self.head(z)
        return logits