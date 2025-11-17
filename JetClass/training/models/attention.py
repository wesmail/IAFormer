# Generic imports
import math

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Flash Attention import
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Warning: flash-attn not installed. Using standard attention.")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, u_embed=6, num_heads=1, dropout=0.0, use_flash=True, top_k=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_flash = use_flash and FLASH_AVAILABLE
        self.top_k = top_k  # NEW: Number of top edges to keep per particle (None = keep all)

        self.u_proj = nn.Conv2d(u_embed, self.num_heads, kernel_size=(1, 1))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout if self.training else 0.0

    def forward(self, x, u=None, umask=None):
        # x shape = (batch, num_tokens, embed_dim)
        bs, num_tokens, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(bs, num_tokens, self.num_heads, self.head_dim)
        K = K.view(bs, num_tokens, self.num_heads, self.head_dim)
        V = V.view(bs, num_tokens, self.num_heads, self.head_dim)

        # Use Flash Attention if available and no custom bias (u) is provided
        if self.use_flash and u is None:
            # Flash Attention expects (batch, seqlen, nheads, headdim)
            # dropout_p should be 0 during eval
            context_vec = flash_attn_func(
                Q, K, V,
                dropout_p=self.dropout_p,
                softmax_scale=self.scaling,
                causal=False
            )
            # context_vec shape: (batch, num_tokens, num_heads, head_dim)
            context_vec = context_vec.contiguous().view(bs, num_tokens, self.embed_dim)
            context_vec = self.out_proj(context_vec)
            
            # Flash attention doesn't return weights, so we return None
            return context_vec, None, None
        
        else:
            # Fall back to standard attention when u (interaction matrix) is provided
            # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
            K = K.transpose(1, 2)
            Q = Q.transpose(1, 2)
            V = V.transpose(1, 2)

            # Compute scaled dot-product attention
            attn_scores = Q @ K.transpose(2, 3)
            attn_scores = attn_scores * self.scaling

            # Add interaction matrix if provided
            if u is not None:
                u = self.u_proj(u)
                
                # NEW: Apply top-k sparsification if enabled
                if self.top_k is not None and self.top_k < num_tokens:
                    # Compute edge importance from interaction matrix
                    # u shape: [B, num_heads, P, P]
                    edge_importance = u.abs().sum(dim=1)  # [B, P, P] - sum across heads
                    
                    # Get top-k neighbors for each particle
                    k = min(self.top_k, num_tokens - 1)  # Don't exceed available tokens
                    topk_values, topk_indices = torch.topk(
                        edge_importance,
                        k=k,
                        dim=-1,
                        largest=True
                    )  # [B, P, k]
                    
                    # Create sparse mask: True for top-k edges, False otherwise
                    sparse_mask = torch.zeros(bs, num_tokens, num_tokens, 
                                             dtype=torch.bool, device=u.device)
                    
                    # Use scatter with flattened indices to avoid broadcasting issues
                    batch_size_actual = topk_indices.shape[0]
                    num_tokens_actual = topk_indices.shape[1]
                    k_actual = topk_indices.shape[2]
                    
                    # Create indices for scatter
                    batch_idx = torch.arange(batch_size_actual, device=u.device).view(-1, 1, 1).expand(-1, num_tokens_actual, k_actual)
                    token_idx = torch.arange(num_tokens_actual, device=u.device).view(1, -1, 1).expand(batch_size_actual, -1, k_actual)
                    
                    # Flatten for proper indexing
                    batch_idx_flat = batch_idx.reshape(-1)
                    token_idx_flat = token_idx.reshape(-1)
                    topk_idx_flat = topk_indices.reshape(-1)
                    
                    # Set top-k positions to True
                    sparse_mask[batch_idx_flat, token_idx_flat, topk_idx_flat] = True
                    
                    # Apply mask to u (set non-top-k edges to large negative value)
                    # Use -1e9 instead of -inf to prevent NaN in softmax
                    sparse_mask_expanded = sparse_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
                    u = u.masked_fill(~sparse_mask_expanded, -1e9)
                
                # Apply umask if provided (for padding)
                if umask is not None:
                    u = u.masked_fill(umask.transpose(3, 1), -1e9)
                
                attn_scores = attn_scores + u

            attn_weights = torch.softmax(attn_scores, dim=-1)
            # Replace NaN with 0 (can happen with all -inf in a row)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            out = self.dropout(attn_weights)

            # Shape: (b, num_tokens, num_heads, head_dim)
            context_vec = (out @ V).transpose(1, 2)

            # Combine heads
            context_vec = context_vec.contiguous().view(bs, num_tokens, self.embed_dim)
            context_vec = self.out_proj(context_vec)

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
        use_flash=True,
        top_k=None,  # NEW
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = num_layers
        self.use_flash = use_flash and FLASH_AVAILABLE
        self.top_k = top_k  # NEW: Number of top edges to keep per particle

        # Dimension per head (2 attention maps for differential attention)
        self.head_dim = embed_dim // num_heads // 2
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
        
        self.dropout_p = dropout if self.training else 0.0

    def forward(self, x, u, umask=None):
        batch_size, num_tokens, _ = x.size()

        v = self.v_proj(x)
        v = v.view(batch_size, num_tokens, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        # u: [B, P, P, u_embed]  -> u_proj -> [B, 2*num_heads, P, P]
        attn_scores = self.u_proj(u)  # (B, 2H, P, P)

        # --------------------------------------------------------------
        # top-k sparsification (optional)
        # --------------------------------------------------------------
        if self.top_k is not None and self.top_k < num_tokens:
            # Edge importance from raw interaction u: [B, P, P, F]
            edge_importance = u.abs().sum(dim=-1)  # [B, P, P]

            k = min(self.top_k, num_tokens - 1)
            _, topk_indices = torch.topk(
                edge_importance,
                k=k,
                dim=-1,
                largest=True,
            )  # [B, P, k]

            sparse_mask = torch.zeros(
                batch_size, num_tokens, num_tokens,
                dtype=torch.bool,
                device=u.device,
            )

            B_act, P_act, k_act = topk_indices.shape
            batch_idx = torch.arange(B_act, device=u.device).view(-1, 1, 1).expand(-1, P_act, k_act)
            token_idx = torch.arange(P_act, device=u.device).view(1, -1, 1).expand(B_act, -1, k_act)

            sparse_mask[batch_idx.reshape(-1),
                        token_idx.reshape(-1),
                        topk_indices.reshape(-1)] = True

            sparse_mask_expanded = sparse_mask.unsqueeze(1).expand(-1, self.num_heads * 2, -1, -1)
            attn_scores = attn_scores.masked_fill(~sparse_mask_expanded, -1e9)

        # --------------------------------------------------------------
        # Padding / particle mask
        # umask: [B, 1, P, P] or [B, P, P]
        # --------------------------------------------------------------
        if umask is not None:
            if umask.dim() == 3:
                umask = umask.unsqueeze(1)
            umask_expanded = umask.expand(-1, self.num_heads * 2, -1, -1)
            attn_scores = attn_scores.masked_fill(umask_expanded, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # lambda / beta
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        beta = (lambda_1 * self.lambda_init)
        beta = torch.sigmoid(beta)

        attn_weights = attn_weights.view(
            batch_size, self.num_heads, 2, num_tokens, num_tokens
        )
        attn_weights = attn_weights[:, :, 0] - beta * attn_weights[:, :, 1]

        # Weighted sum
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

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
        use_flash=True,
        top_k=None,  # NEW
    ):
        super(ParticleAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        if attn in {"interaction", "plain"}:
            self.pmha = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                u_embed=u_embed,
                use_flash=use_flash,
                top_k=top_k,  # Pass top_k
            )
        elif attn == "diff":
            self.pmha = MultiHeadDiffAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                u_embed=u_embed,
                num_layers=num_layers,
                layer_idx=layer_idx,
                use_flash=use_flash,
                top_k=top_k,  # Pass top_k
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
        attn_output, attn_weights, beta = self.pmha(x, u, umask)
        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h, attn_weights, beta
