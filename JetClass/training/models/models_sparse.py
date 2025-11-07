# Generic imports
import math

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Sparse operations
try:
    from torch_scatter import scatter_softmax, scatter_sum
    SCATTER_AVAILABLE = True
except ImportError:
    SCATTER_AVAILABLE = False
    print("Warning: torch-scatter not installed. Sparse attention unavailable.")

# Flash Attention import
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Warning: flash-attn not installed. Using standard attention.")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, u_embed=6, num_heads=1, dropout=0.0, use_flash=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.use_flash = use_flash and FLASH_AVAILABLE

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
                if umask is not None:
                    u.masked_fill_(umask.transpose(3, 1), -torch.inf)
                attn_scores += u

            attn_weights = torch.softmax(attn_scores, dim=-1)
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = num_layers
        self.use_flash = use_flash and FLASH_AVAILABLE

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

        # Note: Differential attention requires computing two attention maps
        # and taking their difference, which doesn't align well with Flash Attention's
        # optimized kernel. Flash Attention is designed for standard attention.
        # For differential attention, we need explicit attention scores.
        
        # Standard implementation (Flash Attention not applicable here)
        attn_scores = self.u_proj(u)
        if umask is not None:
            attn_scores.masked_fill_(umask.transpose(3, 1), -torch.inf)

        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        beta = (lambda_1 * self.lambda_init)
        beta = F.sigmoid(beta)

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


# ============================================================================
# SPARSE ATTENTION MODULES
# ============================================================================

class SparseMultiHeadAttention(nn.Module):
    """
    Efficient attention using sparse edge representation.
    Instead of dense (B, N, N, F) interaction matrix,
    uses edge_index (2, E) and edge_attr (E, F).
    """
    def __init__(self, embed_dim, edge_dim=6, num_heads=8, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        assert SCATTER_AVAILABLE, "torch-scatter required for sparse attention"
        
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
        
        # Ensure edge_index and edge_attr have matching number of edges
        num_edges_from_index = edge_index.shape[1]
        num_edges_from_attr = edge_attr.shape[0]
        if num_edges_from_index != num_edges_from_attr:
            raise ValueError(
                f"Mismatch between edge_index ({num_edges_from_index} edges) "
                f"and edge_attr ({num_edges_from_attr} edges)"
            )
        
        # Ensure tgt is 1D
        tgt = tgt.squeeze() if tgt.dim() > 1 else tgt
        
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
        
        return out, attn_weights, None  # Return None for beta (only diff attention uses it)


class SparseMultiHeadDiffAttention(nn.Module):
    """
    Sparse version of differential attention for edge-based graphs.
    Computes two separate attention maps and takes their difference.
    """
    def __init__(
        self,
        embed_dim,
        edge_dim=6,
        num_heads=8,
        num_layers=1,
        layer_idx=1,
        dropout=0.0,
    ):
        super().__init__()
        assert SCATTER_AVAILABLE, "torch-scatter required for sparse attention"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = num_layers
        
        # Dimension per head (2 attention maps for differential attention)
        self.head_dim = embed_dim // num_heads // 2
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
        self.scaling = self.head_dim ** -0.5
        
        # Value projection
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Lambda parameters for differential attention
        self.lambda_init = lambda_init_fn(self.depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))
        
        # Edge feature encoder - outputs 2 attention maps per head
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_heads * 2)  # 2 maps per head for diff attention
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Args:
            x: (total_particles, embed_dim) - flattened across batch
            edge_index: (2, num_edges) - [source, target] indices
            edge_attr: (num_edges, edge_dim) - edge features
            batch: (total_particles,) - batch assignment for each particle
        
        Returns:
            out: (total_particles, embed_dim)
            attn_weights: (num_edges, num_heads) - differential attention weights
            beta: scalar - lambda parameter value
        """
        num_particles = x.size(0)
        
        # Project to values only (differential attention uses edge features for Q, K)
        V = self.v_proj(x)  # (total_particles, embed_dim)
        
        # Reshape for multi-head: (N, num_heads, 2 * head_dim)
        V = V.view(num_particles, self.num_heads, 2 * self.head_dim)
        
        # Extract source and target indices
        src, tgt = edge_index[0], edge_index[1]
        
        # Ensure edge_index and edge_attr have matching number of edges
        num_edges_from_index = edge_index.shape[1]
        num_edges_from_attr = edge_attr.shape[0]
        if num_edges_from_index != num_edges_from_attr:
            raise ValueError(
                f"Mismatch between edge_index ({num_edges_from_index} edges) "
                f"and edge_attr ({num_edges_from_attr} edges)"
            )
        
        # Encode edge features to get TWO attention scores per edge per head
        edge_scores = self.edge_encoder(edge_attr)  # (num_edges, num_heads * 2)
        edge_scores = edge_scores.view(-1, self.num_heads, 2)  # (num_edges, num_heads, 2)
        
        # Split into two attention maps
        attn_scores_1 = edge_scores[:, :, 0]  # (num_edges, num_heads)
        attn_scores_2 = edge_scores[:, :, 1]  # (num_edges, num_heads)
        
        # Ensure tgt is 1D and matches the number of edges
        tgt = tgt.squeeze() if tgt.dim() > 1 else tgt
        assert tgt.shape[0] == attn_scores_1.shape[0], \
            f"tgt shape {tgt.shape} doesn't match attn_scores_1 shape {attn_scores_1.shape}"
        
        # Apply softmax separately for each attention map
        attn_weights_1 = scatter_softmax(attn_scores_1, tgt, dim=0)
        attn_weights_2 = scatter_softmax(attn_scores_2, tgt, dim=0)
        
        # Compute lambda (differential weighting)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        beta = torch.sigmoid(lambda_1 * self.lambda_init)
        
        # Differential attention: first map - beta * second map
        attn_weights = attn_weights_1 - beta * attn_weights_2  # (num_edges, num_heads)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values: (num_edges, num_heads, 2 * head_dim)
        V_src = V[src]
        weighted_V = attn_weights.unsqueeze(-1) * V_src
        
        # Aggregate to target nodes: (num_particles, num_heads, 2 * head_dim)
        out = scatter_sum(weighted_V, tgt, dim=0, dim_size=num_particles)
        
        # Reshape and project
        out = out.view(num_particles, self.embed_dim)
        out = self.out_proj(out)
        
        return out, attn_weights, beta


# ============================================================================
# ATTENTION BLOCKS
# ============================================================================

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
        sparse=False,  # NEW: Toggle sparse attention
    ):
        super(ParticleAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.sparse = sparse
        
        if sparse:
            # Sparse attention variants
            if attn in {"interaction", "plain"}:
                self.pmha = SparseMultiHeadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    edge_dim=u_embed,
                )
            elif attn == "diff":
                self.pmha = SparseMultiHeadDiffAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    edge_dim=u_embed,
                    num_layers=num_layers,
                    layer_idx=layer_idx,
                )
            else:
                raise ValueError(
                    f"Invalid attention type: {attn}. Must be 'plain', 'interaction' or 'diff'."
                )
        else:
            # Dense attention variants (original)
            if attn in {"interaction", "plain"}:
                self.pmha = MultiHeadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    u_embed=u_embed,
                    use_flash=use_flash,
                )
            elif attn == "diff":
                self.pmha = MultiHeadDiffAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    u_embed=u_embed,
                    num_layers=num_layers,
                    layer_idx=layer_idx,
                    use_flash=use_flash,
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
    
    def forward(self, x, u=None, umask=None, edge_index=None, edge_attr=None, batch=None):
        """
        Args:
            Dense mode: x, u, umask
            Sparse mode: x, edge_index, edge_attr, batch
        """
        x_res = x
        x = self.norm1(x)
        
        if self.sparse:
            attn_output, attn_weights, beta = self.pmha(x, edge_index, edge_attr, batch)
        else:
            attn_output, attn_weights, beta = self.pmha(x, u, umask)
        
        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h, attn_weights, beta


# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# Torch Metrics imports
from torchmetrics import Accuracy, AUROC

# PyG imports
from torch_geometric.data import Batch

# Your modules
from models.embedding import ParticleEmbedding


class SparseTransformer(nn.Module):
    """
    Transformer using sparse edge representation (PyG format).
    Works with edge_index and edge_attr instead of dense adjacency matrix.
    """
    def __init__(
        self,
        in_channels=7,
        edge_channels=6,
        embed_dim=32,
        num_heads=8,
        num_blocks=6,
        attn="plain",
        num_classes=10,
        use_flash=False,  # Flash attention not applicable with sparse
    ):
        super().__init__()
        if attn not in {"plain", "diff", "interaction"}:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be 'plain', 'interaction' or 'diff'."
            )
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.attn = attn
        self.embed_dim = embed_dim

        # Particle embedding
        self.particle_embed = ParticleEmbedding(input_dim=in_channels)

        # Attention blocks with sparse=True
        self.blocks = nn.ModuleList([
            ParticleAttentionBlock(
                embed_dim=embed_dim,
                u_embed=edge_channels,  # This becomes edge_dim in sparse mode
                num_heads=num_heads,
                attn=attn,
                num_layers=num_blocks,
                layer_idx=i,
                use_flash=False,  # Not used in sparse mode
                sparse=True,  # Enable sparse attention
            )
            for i in range(num_blocks)
        ])

        # Global pooling + classification head
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, batch, return_attn_activations=False):
        """
        Args:
            batch: PyG Batch object with attributes:
                - x: (total_particles, in_channels)
                - edge_index: (2, total_edges)
                - edge_attr: (total_edges, edge_channels)
                - batch: (total_particles,) - batch assignment
            return_attn_activations: If True, return attention weights and activations
        
        Returns:
            logits: (batch_size, num_classes)
            attn_stack: attention weights if requested
            act_stack: activations if requested
            mean_beta: mean beta value for diff attention
        """
        # Extract PyG batch components
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_vec = batch.batch  # (total_particles,) indicating which graph each node belongs to
        
        # Embed particles
        x = self.particle_embed(x)  # (total_particles, embed_dim)
        
        # Collect attention/activations if needed
        if return_attn_activations:
            attn_weights = []
            activations = []
        
        # Collect beta values for logging (only for diff attention)
        beta_values = [] if self.attn == "diff" else None

        # Process through attention blocks
        for block in self.blocks:
            x, attn, beta = block(
                x, 
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch=batch_vec
            )
            
            if self.attn == "diff" and beta is not None:
                beta_values.append(beta)
            
            if return_attn_activations:
                attn_weights.append(attn)
                # For activations, we need to aggregate per graph
                # We'll compute mean activation per node, then reshape
                activations.append(x)
        
        # Global pooling: aggregate particles per graph
        # Use scatter_mean to pool node features by batch assignment
        from torch_geometric.utils import scatter
        
        # x: (total_particles, embed_dim)
        # batch_vec: (total_particles,) with values 0, 1, 2, ... indicating graph membership
        batch_size = batch_vec.max().item() + 1
        x_pooled = scatter(x, batch_vec, dim=0, reduce='mean', dim_size=batch_size)
        # x_pooled: (batch_size, embed_dim)
        
        # Classification
        logits = self.mlp_head(x_pooled)  # (batch_size, num_classes)
        
        # Compute mean beta across all blocks for logging
        mean_beta = None
        if beta_values:
            beta_stack = torch.stack(beta_values)
            mean_beta = beta_stack.mean().item()
        
        if return_attn_activations:
            # Stack attention weights: each is (num_edges, num_heads)
            # For compatibility, we'll keep them as a list or stack
            attn_stack = torch.stack(attn_weights) if attn_weights else None
            
            # For activations: each is (total_particles, embed_dim)
            # Stack across layers and reshape to (batch_size, num_layers, ...)
            # This is more complex with variable particles per graph
            # For simplicity, we'll return per-particle activations
            act_stack = torch.stack(activations) if activations else None
            
            return logits, attn_stack, act_stack, mean_beta
        else:
            return logits, None, None, mean_beta


class JetTaggingModule(LightningModule):
    """
    Updated Lightning Module for sparse PyG-based training.
    """
    def __init__(
        self,
        in_channels: int = 7,
        edge_channels: int = 6,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "plain",
        num_classes: int = 10,
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 1e-4,
        batch_size: int = 64,
        label_smoothing: float = 0.0,
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build the sparse model
        self.model = SparseTransformer(
            in_channels=in_channels,
            edge_channels=edge_channels,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn=attn,
            num_classes=num_classes,
        )

        self.compile_model = compile_model

        # Metrics for multiclass
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

        self.lr = lr
        self.eta_min = eta_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        # For predictions (optional)
        self.test_predictions = []
        self.test_targets = []
        self.attn_weights = []
        self.activations = []

    def configure_model(self):
        """
        Compile model with torch.compile if enabled.
        Note: torch.compile may have issues with dynamic graphs in PyG.
        Consider disabling or using dynamic=True mode.
        """
        if self.compile_model and not hasattr(self, '_compiled'):
            print("Compiling model with torch.compile...")
            
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,  # Must be False for dynamic graphs
                    dynamic=True,  # Handle dynamic shapes in PyG batches
                )
                self._compiled = True
                print("✅ Model compilation complete!")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
                print("Continuing without compilation...")
                self.compile_model = False

    # ---- helpers ----
    def _step(self, batch, prefix: str):
        """
        Args:
            batch: PyG Batch object from DataLoader
        """
        # Extract labels from PyG batch
        y = batch.y  # (batch_size,)
        
        # Forward pass
        logits, _, _, mean_beta = self.model(batch, return_attn_activations=False)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        # Metrics
        acc = self.acc(logits, y)
        probs = logits.softmax(dim=-1)
        auc = self.auroc(probs, y)

        # Optimize logging: sync_dist only when needed
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False

        # Get actual batch size from PyG batch
        actual_batch_size = y.size(0)

        # Reduce logging overhead
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)
        self.log(f"{prefix}_acc",  acc,  prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size, on_step=True, on_epoch=True)
        self.log(f"{prefix}_auc",  auc,  prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size, on_step=True, on_epoch=True)

        # Log beta value if available (for diff attention)
        if mean_beta is not None:
            self.log(f"{prefix}_beta", mean_beta, prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size, on_step=True, on_epoch=True)

        return loss, logits

    # ---- Lightning hooks ----
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        y = batch.y
        
        # For test, we need attention weights and activations
        logits, attn_weights, activations, mean_beta = self.model(
            batch, 
            return_attn_activations=True
        )
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        probs = logits.softmax(dim=-1)
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False
        actual_batch_size = y.size(0)
        
        self.log("test_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)
        self.log("test_acc",  self.acc(logits, y), prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)
        self.log("test_auc",  self.auroc(probs, y), prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)

        # Log beta value if available
        if mean_beta is not None:
            self.log("test_beta", mean_beta, prog_bar=False, sync_dist=sync_dist, batch_size=actual_batch_size)

        # Stash predictions (optional)
        self.test_predictions.extend(probs.detach().cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        # Note: Attention weights and activations have different shapes in sparse mode
        # attn_weights: (num_layers, num_edges, num_heads)
        # activations: (num_layers, total_particles, embed_dim)
        # You may want to process these differently than in dense mode
        if attn_weights is not None:
            # Store per-batch for later analysis
            self.attn_weights.append(attn_weights.detach().cpu())
        if activations is not None:
            self.activations.append(activations.detach().cpu())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=self.eta_min)
        return {"optimizer": opt, "lr_scheduler": sch}


# ============================================================================
# HYBRID MODULE: Supports both dense and sparse modes
# ============================================================================

class HybridJetTaggingModule(LightningModule):
    """
    Hybrid module that supports both dense and sparse modes.
    Set sparse=True to use PyG sparse representation.
    Set sparse=False to use original dense representation.
    """
    def __init__(
        self,
        in_channels: int = 7,
        u_channels: int = 6,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "plain",
        max_num_particles: int = 100,  # Only used in dense mode
        num_classes: int = 10,
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 1e-4,
        batch_size: int = 64,
        label_smoothing: float = 0.0,
        compile_model: bool = True,
        sparse: bool = True,  # Toggle between sparse/dense
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.sparse = sparse

        if sparse:
            # Use sparse transformer
            self.model = SparseTransformer(
                in_channels=in_channels,
                edge_channels=u_channels,
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                attn=attn,
                num_classes=num_classes,
            )
        else:
            # Use original dense transformer
            from models import Transformer  # Your original dense model
            self.model = Transformer(
                in_channels=in_channels,
                u_channels=u_channels,
                embed_dim=embed_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                max_num_particles=max_num_particles,
                attn=attn,
                num_classes=num_classes,
            )

        self.compile_model = compile_model
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

        self.lr = lr
        self.eta_min = eta_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        self.test_predictions = []
        self.test_targets = []
        self.attn_weights = []
        self.activations = []

    def configure_model(self):
        if self.compile_model and not hasattr(self, '_compiled'):
            print("Compiling model with torch.compile...")
            try:
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=self.sparse,  # Enable dynamic for sparse mode
                )
                self._compiled = True
                print("✅ Model compilation complete!")
            except Exception as e:
                print(f"⚠️  Model compilation failed: {e}")
                self.compile_model = False

    def _step(self, batch, prefix: str):
        if self.sparse:
            # PyG batch
            y = batch.y
            logits, _, _, mean_beta = self.model(batch, return_attn_activations=False)
            actual_batch_size = y.size(0)
        else:
            # Dense batch (dict format)
            x, u, y = batch["node_features"], batch["edge_features"], batch["labels"]
            logits, _, _, mean_beta = self.model(x, u, return_attn_activations=False)
            actual_batch_size = y.size(0)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        acc = self.acc(logits, y)
        probs = logits.softmax(dim=-1)
        auc = self.auroc(probs, y)

        sync_dist = self.trainer.num_devices > 1 if self.trainer else False
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)
        self.log(f"{prefix}_acc",  acc,  prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size, on_step=False, on_epoch=True)
        self.log(f"{prefix}_auc",  auc,  prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size, on_step=False, on_epoch=True)

        if mean_beta is not None:
            self.log(f"{prefix}_beta", mean_beta, prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size, on_step=False, on_epoch=True)

        return loss, logits

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        if self.sparse:
            y = batch.y
            logits, attn_weights, activations, mean_beta = self.model(batch, return_attn_activations=True)
        else:
            x, u, y = batch["node_features"], batch["edge_features"], batch["labels"]
            logits, attn_weights, activations, mean_beta = self.model(x, u, return_attn_activations=True)
        
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
        probs = logits.softmax(dim=-1)
        
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False
        actual_batch_size = y.size(0)
        
        self.log("test_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)
        self.log("test_acc",  self.acc(logits, y), prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)
        self.log("test_auc",  self.auroc(probs, y), prog_bar=True, sync_dist=sync_dist, batch_size=actual_batch_size)

        if mean_beta is not None:
            self.log("test_beta", mean_beta, prog_bar=False, sync_dist=sync_dist, batch_size=actual_batch_size)

        self.test_predictions.extend(probs.detach().cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        
        if attn_weights is not None:
            self.attn_weights.append(attn_weights.detach().cpu())
        if activations is not None:
            self.activations.append(activations.detach().cpu())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=self.eta_min)
        return {"optimizer": opt, "lr_scheduler": sch}        