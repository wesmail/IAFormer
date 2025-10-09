# Generic imports
import math

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, u_embed=6, num_heads=1, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = (
            embed_dim // num_heads
        )  # Reduce the projection dim to match desired output dim
        self.scaling = self.head_dim**-0.5

        self.u_proj = nn.Conv2d(u_embed, self.num_heads, kernel_size=(1, 1))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Linear output layer to combine head outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, u=None, umask=None):
        # x shape = (batch, num_tokens, embed_dim)
        bs, num_tokens, _ = x.shape

        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        K = K.view(bs, num_tokens, self.num_heads, self.head_dim)
        V = V.view(bs, num_tokens, self.num_heads, self.head_dim)
        Q = Q.view(bs, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = Q @ K.transpose(
            2, 3
        )  # Dot product for each head (num_tokens, head_dim) * (head_dim, num_tokens)
        attn_scores = attn_scores / self.scaling

        # Change feature dimention in the interaction matrix
        if u is not None:
            u = self.u_proj(u)
            if umask is not None:
                u.masked_fill_(umask.transpose(3, 1), -torch.inf)

            attn_scores += u

        attn_weights = torch.softmax(attn_scores, dim=-1)
        out = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (out @ V).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(bs, num_tokens, self.embed_dim)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec, attn_weights, None


def lambda_init_fn(depth):
    return 1 - math.exp(-1 / depth)


class MultiHeadDiffAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        u_embed=6,
        num_heads=1,
        num_layers=1,
        layer_idx=1,
        dropout=0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = num_layers

        # Dimension per head
        self.head_dim = embed_dim // num_heads // 2  # 2 attention maps
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
        self.scaling = self.head_dim**-0.5

        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Lambda parameters
        self.lambda_init = lambda_init_fn(self.depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))

        self.u_proj = nn.Conv2d(u_embed, self.num_heads * 2, kernel_size=(1, 1))

    def forward(self, x, u, umask=None):
        batch_size, num_tokens, _ = x.size()

        v = self.v_proj(x)
        v = v.view(batch_size, num_tokens, self.num_heads, 2 * self.head_dim).transpose(
            1, 2
        )

        attn_scores = self.u_proj(u)
        if umask is not None:
            attn_scores.masked_fill_(umask.transpose(3, 1), -torch.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        # Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        beta = (lambda_1 * self.lambda_init).clamp_(max=1)

        attn_weights = attn_weights.view(
            batch_size, self.num_heads, 2, num_tokens, num_tokens
        )
        attn_weights = attn_weights[:, :, 0] - beta * attn_weights[:, :, 1]

        # Weighted sum
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

        # Final projection
        out = self.out_proj(out)

        return out, attn_weights, beta


class ParticleAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        u_embed,
        expansion_factor=4,
        num_heads=1,
        attn="plain",
        num_layers=1,
        layer_idx=1,
    ):
        super(ParticleAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if attn in {"interaction", "plain"}:
            self.pmha = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                u_embed=u_embed,
            )
        elif attn == "diff":
            self.pmha = MultiHeadDiffAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                u_embed=u_embed,
                num_layers=num_layers,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be 'plain', 'interaction' or 'diff'."
            )
        self.mlp = nn.Sequential(
            nn.RMSNorm(embed_dim),
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.SiLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )
    def forward(self, x, u=None, umask=None):
        x_res = x
        x = self.norm1(x)
        attn_output, attn_weights, beta = self.pmha(x, u, umask)  # x, and u embeddings
        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h, attn_weights, beta
