import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


def lambda_init_fn_layer_idx(layer_idx, depth):
    decay = -0.3 * (28 / depth)  # 28 is total layers in the paper
    return 0.8 - 0.6 * math.exp(decay * (layer_idx - 1))


class ParticleEmbedding(nn.Module):
    def __init__(self, input_dim=11):
        super(ParticleEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


class InteractionInputEncoding(nn.Module):
    def __init__(self, input_dim=4, output_dim=8):
        super(InteractionInputEncoding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        return x.permute(0, 2, 3, 1)


class MultiheadDiffAttn(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, layer_idx=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Dimension per head
        self.head_dim = embed_dim // num_heads // 2  # 2 attention maps
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
        self.scaling = self.head_dim**-0.5

        # Projections
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.u_proj = nn.Conv2d(64, self.num_heads * 2, kernel_size=(1, 1))

        # Lambda parameters
        self.lambda_init = lambda_init_fn_layer_idx(layer_idx=layer_idx, depth=depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.embed_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.embed_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.embed_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.embed_dim).normal_(mean=0, std=0.1)
        )

        # Normalization layer
        self.subln = nn.LayerNorm(self.head_dim * 2)

    def forward(self, x, u=None, umask=None):
        batch_size, num_tokens, _ = x.size()

        # Linear projections
        v = self.v_proj(x)

        # Reshape into heads
        v = v.view(batch_size, num_tokens, self.num_heads, 2 * self.head_dim).transpose(
            1, 2
        )

        # Compute attention weights
        u = self.u_proj(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if umask is not None:
            u.masked_fill_(umask, -torch.inf)
            attn_weights = u.transpose(3, 1)

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Apply differential attention
        attn_weights = attn_weights.view(
            batch_size, self.num_heads, 2, num_tokens, num_tokens
        )
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        # Weighted sum
        attn = torch.matmul(attn_weights, v)

        # Normalize and reshape
        attn = self.subln(attn)
        attn = attn.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

        # Final projection
        attn = self.out_proj(attn)
        return attn


class ParticleAttentionBlock(nn.Module):
    def __init__(
        self, embed_dim, expansion_factor=4, num_heads=1, num_layers=1, layer_idx=1
    ):
        super(ParticleAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pmha = MultiheadDiffAttn(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=num_layers,
            layer_idx=layer_idx,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.LayerNorm(expansion_factor * embed_dim),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

    def forward(self, x, u=None, umask=None):
        x_res = x
        x = self.norm1(x)
        attn_output = self.pmha(x, u, umask)  # x, and u embeddings
        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h
