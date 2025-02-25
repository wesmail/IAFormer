# Torch imports
import torch.nn as nn


class ParticleEmbedding(nn.Module):
    def __init__(self, input_dim=11):
        super(ParticleEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.RMSNorm(32),
        )

    def forward(self, x):
        return self.mlp(x)


class InteractionInputEncoding(nn.Module):
    def __init__(self, input_dim=4, output_dim=64):
        super(InteractionInputEncoding, self).__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(256, 128, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(128, output_dim, kernel_size=(1, 1)),
            nn.GELU(),
        )

    def forward(self, x):
        return self.embed(x.transpose(3, 1))
