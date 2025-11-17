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
from torch_geometric.nn import global_mean_pool

# Generic imports
import math

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports (for sparse attention)
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Flash Attention import (not actually used for diff attention, but kept for API compatibility)
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Warning: flash-attn not installed. Using standard attention.")


def lambda_init_fn(depth: int) -> float:
    return 1 - math.exp(-1 / depth)

class ParticleEmbedding(nn.Module):
    def __init__(self, input_dim=11):
        super(ParticleEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.RMSNorm(64),
        )

    def forward(self, x):
        return self.mlp(x)

# ---------------------------------------------------------------------------
# Edge interaction encoding (sparse analogue of your InteractionInputEncoding)
# ---------------------------------------------------------------------------
class EdgeInteractionEncoding(nn.Module):
    """
    MLP-based encoder for edge features, analogous to your Conv2d-based
    InteractionInputEncoding for dense interaction matrices.

    Input:
        edge_attr: [E, input_dim]
    Output:
        edge_u:    [E, output_dim]
    """
    def __init__(self, input_dim: int = 4, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.GELU(),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        # edge_attr: [E, input_dim]
        return self.net(edge_attr)  # [E, output_dim]


# ---------------------------------------------------------------------------
# Sparse Multi-Head Differential Attention
# ---------------------------------------------------------------------------
class MultiHeadDiffAttention(MessagePassing):
    """
    PyG-native differential multi-head attention on sparse graphs.

    This is the sparse analogue of your dense MultiHeadDiffAttention that used
    Conv2d over a dense interaction matrix u[b, :, i, j].

    Here we operate on:
        x:          [N, embed_dim]         (node features)
        edge_index: [2, E]                 (source -> target)
        u:          [E, u_embed]           (encoded edge features, e.g. via EdgeInteractionEncoding)

    Internals:
        - v = W_v x           -> [N, H, 2*Dh]
        - scores = W_u(e_ij)  -> [E, H, 2]  (two attention maps per head)
        - α1, α2 via segment softmax over neighbors per target node
        - α = α1 - β α2       (β is a learned scalar from λ parameters)
        - out_i = sum_j α_ij * v_j    (per head, then merged)
    """

    def __init__(
        self,
        embed_dim: int,
        u_embed: int,
        num_heads: int = 1,
        num_layers: int = 1,  # for lambda_init
        layer_idx: int = 1,   # kept for API symmetry, not used explicitly
        dropout: float = 0.0,
        use_flash: bool = True,  # kept for API compatibility (not used)
    ):
        # node_dim=0 so MessagePassing knows where node features live
        super().__init__(node_dim=0, aggr="add")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = num_layers
        self.use_flash = use_flash and FLASH_AVAILABLE

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"

        # 2 attention maps per head → half the per-head dim for each
        self.head_dim = embed_dim // num_heads // 2
        self.scaling = self.head_dim ** -0.5  # kept for possible Q/K extensions

        # Project node features
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Project encoded edge features into 2*num_heads attention scores per edge
        # (two maps per head)
        self.u_proj = nn.Linear(u_embed, 2 * num_heads, bias=False)

        # Lambda / beta parameters (same logic as your dense implementation)
        self.lambda_init = lambda_init_fn(self.depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # [N, embed_dim]
        edge_index: torch.Tensor, # [2, E]
        u: torch.Tensor,          # [E, u_embed]  (encoded edge features)
        umask: torch.Tensor = None,  # kept for API compatibility, unused in sparse setting
    ):
        N, d_model = x.size()
        assert d_model == self.embed_dim, "x.size(1) must equal embed_dim"

        # Project node features
        v = self.v_proj(x)  # [N, embed_dim]
        v = v.view(N, self.num_heads, 2 * self.head_dim)  # [N, H, 2*Dh]

        # Project edge features to two attention maps per head
        # scores: [E, 2H] -> [E, H, 2]
        scores = self.u_proj(u)           # [E, 2*H]
        scores = scores.view(-1, self.num_heads, 2)  # [E, H, 2]

        # Compute β (global scalar as in your dense version)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        beta = (lambda_1 * self.lambda_init)
        beta = torch.sigmoid(beta)  # scalar in (0,1)
        beta = beta.to(x.device)

        # Normalize attention scores per target node (edge_index[1])
        # Map 1
        alpha1 = softmax(
            scores[..., 0],        # [E, H]
            edge_index[1],         # normalize per dst node
        )
        # Map 2
        alpha2 = softmax(
            scores[..., 1],        # [E, H]
            edge_index[1],
        )

        # Differential attention α = α1 - β α2
        alpha = alpha1 - beta * alpha2  # [E, H]

        # Optional dropout on attention
        alpha = self.dropout(alpha)

        # MessagePassing: propagate v from source to target with weights alpha
        # We pass v as x to MessagePassing so we can access v_j at edges
        out = self.propagate(
            edge_index,
            x=v,         # v: [N, H, 2*Dh]
            alpha=alpha  # [E, H]
        )  # -> [N, H, 2*Dh] after aggregation

        # Merge heads
        out = out.reshape(N, self.embed_dim)  # [N, embed_dim]
        out = self.out_proj(out)

        # For logging, return alpha and beta
        # alpha: [E, H], beta: scalar tensor
        return out, alpha, beta

    def message(self, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        x_j:   [E, H, 2*Dh]  (source node features for each edge)
        alpha: [E, H]        (attention coefficient per edge & head)
        """
        return alpha.unsqueeze(-1) * x_j  # [E, H, 2*Dh]


# ---------------------------------------------------------------------------
# ParticleAttentionBlock (now supports sparse diff attention via edge_index/edge_attr)
# ---------------------------------------------------------------------------
class ParticleAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        u_embed: int,
        expansion_factor: float = 4.0,
        num_heads: int = 1,
        attn: str = "plain",      # "plain", "interaction", or "diff"
        num_layers: int = 1,
        layer_idx: int = 1,
        use_flash: bool = True,
        edge_in_dim: int = None,  # NEW: raw edge_attr dim (required for attn="diff")
    ):
        """
        For dense 'plain' / 'interaction' attention:
            - We assume your existing MultiHeadAttention works on dense u matrices.

        For sparse 'diff' attention (PyG Data-style):
            - edge_in_dim: input dimension of edge_attr
            - u_embed: encoded edge dimension (output of EdgeInteractionEncoding)

        """
        super(ParticleAttentionBlock, self).__init__()
        self.attn_type = attn
        self.embed_dim = embed_dim

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # -----------------------------
        # Attention module selection
        # -----------------------------
        if attn in {"interaction", "plain"}:
            raise NotImplementedError("Dense attention is not implemented for sparse mode.")

        elif attn == "diff":
            # Sparse differential attention
            if edge_in_dim is None:
                raise ValueError(
                    "edge_in_dim must be provided for 'diff' attention to encode edge_attr."
                )

            # Edge encoder: analogous to your InteractionInputEncoding, but for edge_attr
            self.edge_encoder = EdgeInteractionEncoding(
                input_dim=edge_in_dim,
                output_dim=u_embed,
            )

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

        # -----------------------------
        # MLP block
        # -----------------------------
        self.mlp = nn.Sequential(
            nn.RMSNorm(embed_dim),
            nn.Linear(embed_dim, int(expansion_factor * embed_dim)),
            nn.SiLU(),
            nn.Linear(int(expansion_factor * embed_dim), embed_dim),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor = None,
        umask: torch.Tensor = None,
        edge_index: torch.Tensor = None,
        edge_attr: torch.Tensor = None,
    ):
        """
        Args:
            x:          node features
                - dense case: [B, P, embed_dim]
                - sparse case: [N, embed_dim] (PyG Batch)

            For dense "plain"/"interaction":
                u:      dense interaction tensor (e.g. [B, u_embed, P, P])
                umask:  optional mask

            For sparse "diff":
                edge_index: [2, E]
                edge_attr:  [E, edge_in_dim]

        Returns:
            out:          updated node features
            attn_weights: attention weights (format depends on attention type)
            beta:         scalar (for diff attention), else None
        """
        x_res = x
        x = self.norm1(x)

        if self.attn_type == "diff":
            if edge_index is None or edge_attr is None:
                raise ValueError(
                    "For 'diff' attention, edge_index and edge_attr must be provided."
                )
            if self.edge_encoder is None:
                raise RuntimeError("edge_encoder is not initialized for 'diff' attention.")

            # Encode edge_attr once per block (sparse analogue of InteractionInputEncoding)
            u_encoded = self.edge_encoder(edge_attr)  # [E, u_embed]

            attn_output, attn_weights, beta = self.pmha(
                x, edge_index, u_encoded, umask=None
            )
        else:
            # Dense attention path (unchanged; uses dense u)
            attn_output, attn_weights, beta = self.pmha(x, u, umask)

        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h, attn_weights, beta


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 7,      # node feature dim
        u_channels: int = 6,       # edge_attr dim
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "diff",        # "plain", "diff", "interaction"
        num_classes: int = 10,
        use_flash: bool = True,
    ):
        super().__init__()
        if attn not in {"plain", "diff", "interaction"}:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be 'plain', 'interaction' or 'diff'."
            )
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.attn = attn
        self.u_embed_dim = 64  # encoded edge dim (for diff / interaction if needed)

        # Node embedding
        self.particle_embed = ParticleEmbedding(input_dim=in_channels)

        # For sparse "diff" attention, InteractionInputEncoding is replaced by
        # an edge MLP inside ParticleAttentionBlock (EdgeInteractionEncoding).
        # For dense "plain"/"interaction", your existing ParticleAttentionBlock
        # still expects u_embed, but we won't use edges here unless you extend it.

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if attn == "diff":
                # Sparse differential attention: we must pass edge_in_dim=u_channels
                block = ParticleAttentionBlock(
                    embed_dim=embed_dim,
                    u_embed=self.u_embed_dim,
                    num_heads=num_heads,
                    attn="diff",
                    num_layers=num_blocks,
                    layer_idx=i,
                    use_flash=use_flash,
                    edge_in_dim=u_channels,  # <- raw edge_attr dim
                )
            else:
                # "plain" or "interaction" (dense-style, no sparse edges used yet)
                block = ParticleAttentionBlock(
                    embed_dim=embed_dim,
                    u_embed=self.u_embed_dim,
                    num_heads=num_heads,
                    attn=attn,
                    num_layers=num_blocks,
                    layer_idx=i,
                    use_flash=use_flash,
                    # edge_in_dim not needed in dense mode
                )
            self.blocks.append(block)

        # Graph-level head: pool to [B, embed_dim] then linear
        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, data, return_attn_activations: bool = False):
        """
        Args:
            data: PyG Data/Batch object with
                - data.x          [N, in_channels]
                - data.edge_index [2, E]
                - data.edge_attr  [E, u_channels]
                - data.batch      [N] (graph indices)
                - data.y          [B] (labels)
            return_attn_activations: if True, return attention weights & activations.
        """
        x = data.x                      # [N, in_channels]
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        batch_idx = getattr(data, "batch", None)

        if batch_idx is None:
            # single graph case: all nodes belong to graph 0
            batch_idx = x.new_zeros(x.size(0), dtype=torch.long)

        # 1) Node embedding
        x = self.particle_embed(x)      # [N, embed_dim]

        # For collecting attention / activations
        attn_weights_all = [] if return_attn_activations else None
        activations_all = [] if return_attn_activations else None

        # Collect beta values for logging (only for diff attention)
        beta_values = [] if self.attn == "diff" else None

        # 2) Stack attention blocks
        for block in self.blocks:
            if self.attn == "diff":
                # Sparse diff attention: use edge_index & edge_attr
                x, attn, beta = block(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
                if beta_values is not None and beta is not None:
                    beta_values.append(
                        beta if torch.is_tensor(beta) else torch.tensor(beta, device=x.device)
                    )
            elif self.attn == "plain":
                # Plain self-attention: ignore edges (node-only)
                x, attn, _ = block(x)
            else:  # "interaction" placeholder (dense interaction not wired to sparse yet)
                # You can later build a dense u from edge_index/edge_attr and pass it here.
                raise NotImplementedError(
                    "attn='interaction' with sparse edges is not implemented yet."
                )

            if return_attn_activations:
                attn_weights_all.append(attn)
                # e.g., mean over feature dim per node as a cheap "activation"
                activations_all.append(x.mean(dim=-1))  # [N]

        # 3) Graph pooling + classification head
        # x: [N, embed_dim] -> [B, embed_dim]
        x_graph = global_mean_pool(x, batch_idx)  # mean over nodes per graph
        logits = self.mlp_head(x_graph)           # [B, num_classes]

        # 4) Mean beta (for diff attention logging)
        mean_beta = None
        if beta_values:
            beta_stack = torch.stack(beta_values)  # [num_blocks]
            mean_beta = beta_stack.mean().item()

        if return_attn_activations:
            # Note: for diff attention, attn is [E, H]; stacking => [L, E, H]
            # Transpose to [Batches?] if you later want a different layout.
            attn_stack = torch.stack(attn_weights_all)  # [L, ...]
            act_stack = torch.stack(activations_all)    # [L, N]
            return logits, attn_stack, act_stack, mean_beta
        else:
            return logits, None, None, mean_beta


class JetTaggingModule(LightningModule):
    def __init__(
        self,
        in_channels: int = 7,
        u_channels: int = 6,     # edge_attr dim
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "diff",      # use "diff" to exploit sparsity
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

        # Build the model (sparse / dense depending on attn)
        self.model = Transformer(
            in_channels=in_channels,
            u_channels=u_channels,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn=attn,
            num_classes=num_classes,
        )

        # torch.compile flag
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

        # for predictions (optional)
        self.test_predictions = []
        self.test_targets = []
        self.attn_weights = []
        self.activations = []

    #def configure_model(self):
        """
        Lightning calls this hook at the right time
        (after model is moved to device, before training starts).
        """
    #    if self.compile_model and not hasattr(self, "_compiled"):
    #        print("Compiling model with torch.compile...")
    #        self.model = torch.compile(
    #            self.model,
    #            mode="reduce-overhead",
    #            fullgraph=False,
    #        )
    #        self._compiled = True
    #        print("✅ Model compilation complete!")

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

        # Metrics - detach to prevent memory leaks
        with torch.no_grad():
            acc = self.acc(logits.detach(), y)
            probs = logits.detach().softmax(dim=-1)
            auc = self.auroc(probs, y)

        # Optimize logging: sync_dist only when needed
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False

        # Get actual batch size from PyG batch
        actual_batch_size = y.size(0)

        # CRITICAL FIX: Detach all logged values and reduce on_step logging
        # Only log loss on_step for training monitoring
        is_train = prefix == "train"
        
        self.log(
            f"{prefix}_loss", 
            loss.detach(),  # ✅ DETACH to prevent memory leak
            prog_bar=True, 
            sync_dist=sync_dist, 
            batch_size=actual_batch_size,
            on_step=is_train,  # Only log steps during training
            on_epoch=True
        )
        self.log(
            f"{prefix}_acc", 
            acc.detach(),  # ✅ DETACH to prevent memory leak
            prog_bar=True, 
            sync_dist=sync_dist, 
            batch_size=actual_batch_size, 
            on_step=False,  # ✅ DISABLED on_step to reduce overhead
            on_epoch=True
        )
        self.log(
            f"{prefix}_auc", 
            auc.detach(),  # ✅ DETACH to prevent memory leak
            prog_bar=False,  # Less critical, don't show in progress bar
            sync_dist=sync_dist, 
            batch_size=actual_batch_size, 
            on_step=False,  # ✅ DISABLED on_step to reduce overhead
            on_epoch=True
        )

        # Log beta value if available (for diff attention)
        if mean_beta is not None:
            self.log(
                f"{prefix}_beta", 
                mean_beta.detach() if isinstance(mean_beta, torch.Tensor) else mean_beta,  # ✅ DETACH
                prog_bar=False, 
                sync_dist=sync_dist, 
                batch_size=actual_batch_size, 
                on_step=False,  # ✅ DISABLED on_step
                on_epoch=True
            )

        return loss, logits
    
    # ---- Lightning hooks ----
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # For test, we can optionally collect attention/activations
        logits, attn_weights, activations, mean_beta = self.model(
            batch, return_attn_activations=True
        )
        y = batch.y
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        probs = logits.softmax(dim=-1)
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False

        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            sync_dist=sync_dist,
            batch_size=self.batch_size,
        )
        self.log(
            "test_acc",
            self.acc(logits, y),
            prog_bar=False,
            sync_dist=sync_dist,
            batch_size=self.batch_size,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test_auc",
            self.auroc(probs, y),
            prog_bar=False,
            sync_dist=sync_dist,
            batch_size=self.batch_size,
            on_step=False,
            on_epoch=True,
        )

        if mean_beta is not None:
            self.log(
                "test_beta",
                mean_beta,
                prog_bar=False,
                sync_dist=sync_dist,
                batch_size=self.batch_size,
                on_step=False,
                on_epoch=True,
            )

        # stash (optional)
        self.test_predictions.extend(probs.detach().cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        if attn_weights is not None:
            self.attn_weights.extend(attn_weights.detach().cpu().numpy())
        if activations is not None:
            self.activations.extend(activations.detach().cpu().numpy())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
        }

