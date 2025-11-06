import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum
from torch_geometric.nn import global_mean_pool

class SparseMultiHeadAttention(nn.Module):
    """
    Efficient attention using sparse edge representation.
    Instead of dense (B, N, N, F) interaction matrix,
    uses edge_index (2, E) and edge_attr (E, F).
    """
    def __init__(self, embed_dim, edge_dim=6, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        
        # Standard Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Edge feature encoder (replaces u_proj)
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_heads)
        )
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Args:
            x: (total_particles, embed_dim) - flattened across batch
            edge_index: (2, num_edges) - [source, target] indices
            edge_attr: (num_edges, edge_dim) - edge features
            batch: (total_particles,) - batch assignment for each particle
        """
        num_particles = x.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (total_particles, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head: (N, num_heads, head_dim)
        Q = Q.view(num_particles, self.num_heads, self.head_dim)
        K = K.view(num_particles, self.num_heads, self.head_dim)
        V = V.view(num_particles, self.num_heads, self.head_dim)
        
        # Extract source and target indices
        src, tgt = edge_index[0], edge_index[1]
        
        # Compute attention scores for edges only
        # Q_tgt: (num_edges, num_heads, head_dim)
        Q_tgt = Q[tgt]  
        K_src = K[src]
        
        # Scaled dot product: (num_edges, num_heads)
        attn_scores = (Q_tgt * K_src).sum(dim=-1) * self.scaling
        
        # Add edge bias from interaction features
        edge_bias = self.edge_encoder(edge_attr)  # (num_edges, num_heads)
        attn_scores = attn_scores + edge_bias
        
        # Softmax per target node using scatter operations
        # This is the sparse equivalent of softmax over source nodes
        attn_weights = scatter_softmax(attn_scores, tgt, dim=0)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values: (num_edges, num_heads, head_dim)
        V_src = V[src]
        weighted_V = attn_weights.unsqueeze(-1) * V_src
        
        # Aggregate to target nodes: (num_particles, num_heads, head_dim)
        out = scatter_sum(weighted_V, tgt, dim=0, dim_size=num_particles)
        
        # Reshape and project
        out = out.view(num_particles, self.embed_dim)
        out = self.out_proj(out)
        
        return out, attn_weights


class SparseParticleAttentionBlock(nn.Module):
    def __init__(self, embed_dim, edge_dim=6, num_heads=8, expansion_factor=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.attn = SparseMultiHeadAttention(
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            num_heads=num_heads
        )
        
        self.mlp = nn.Sequential(
            nn.RMSNorm(embed_dim),
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.SiLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        x_res = x
        x = self.norm1(x)
        attn_out, attn_weights = self.attn(x, edge_index, edge_attr, batch)
        x = self.norm2(attn_out)
        h = x + x_res
        z = self.mlp(h)
        return z + h, attn_weights

# Usage in your model
class SparseTransformer(nn.Module):
    def __init__(self, in_channels=11, embed_dim=64, edge_dim=6, num_heads=8, num_blocks=6):
        super().__init__()
        
        from models.embedding import ParticleEmbedding
        self.particle_embed = ParticleEmbedding(input_dim=in_channels)
        
        self.blocks = nn.ModuleList([
            SparseParticleAttentionBlock(
                embed_dim=embed_dim,
                edge_dim=edge_dim,
                num_heads=num_heads
            )
            for _ in range(num_blocks)
        ])
        
        self.mlp_head = nn.Linear(embed_dim, 10)  # num_classes
    
    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: (total_particles, features)
            edge_index: (2, num_edges) - [source, target] indices
            edge_attr: (num_edges, edge_dim) - edge features
            batch: (total_particles,) - batch assignment for each particle
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # x is already flattened across graphs in PyG: (num_nodes_total, feat)
        x = self.particle_embed(x)

        for block in self.blocks:
            x, _ = block(x, edge_index, edge_attr, batch)

        # Pool per graph using the batch vector
        x_pooled = global_mean_pool(x, batch)

        logits = self.mlp_head(x_pooled)
        return logits