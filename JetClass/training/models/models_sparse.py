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
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import softmax

# Generic imports
import math

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
# Edge interaction encoding (sparse analogue of InteractionInputEncoding)
# ---------------------------------------------------------------------------
class EdgeInteractionEncoding(nn.Module):
    """
    MLP-based encoder for edge features, analogous to Conv2d-based
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

    This is the sparse analogue of dense MultiHeadDiffAttention that used
    Conv2d over a dense interaction matrix u[b, :, i, j].

    Here we operate on:
        x:          [N, embed_dim]         (node features)
        edge_index: [2, E]                 (source -> target)
        u:          [E, u_embed]           (encoded edge features)

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
        layer_idx: int = 1,   # kept for API symmetry
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
        self.scaling = self.head_dim ** -0.5

        # Project node features
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Project encoded edge features into 2*num_heads attention scores per edge
        self.u_proj = nn.Linear(u_embed, 2 * num_heads, bias=False)

        # Lambda / beta parameters
        self.lambda_init = lambda_init_fn(self.depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(100).normal_(mean=0, std=0.1))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,          # [N, embed_dim]
        edge_index: torch.Tensor, # [2, E]
        u: torch.Tensor,          # [E, u_embed]  (encoded edge features)
        umask: torch.Tensor = None,  # kept for API compatibility
    ):
        N, d_model = x.size()
        assert d_model == self.embed_dim, "x.size(1) must equal embed_dim"

        # Project node features
        v = self.v_proj(x)  # [N, embed_dim]
        v = v.view(N, self.num_heads, 2 * self.head_dim)  # [N, H, 2*Dh]

        # Project edge features to two attention maps per head
        scores = self.u_proj(u)           # [E, 2*H]
        scores = scores.view(-1, self.num_heads, 2)  # [E, H, 2]

        # 🔥 OPTIMIZATION: Compute β without gradients when possible
        with torch.no_grad():
            lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
            beta = torch.sigmoid(lambda_1 * self.lambda_init)  # scalar in (0,1)
        
        # Convert to tensor on correct device for differentiation if needed
        beta = beta.to(x.device)
        if self.training:
            # Re-enable gradients for training
            beta = beta.detach().requires_grad_(True)

        # Normalize attention scores per target node (edge_index[1])
        alpha1 = softmax(scores[..., 0], edge_index[1])  # [E, H]
        alpha2 = softmax(scores[..., 1], edge_index[1])  # [E, H]

        # Differential attention α = α1 - β α2
        alpha = alpha1 - beta * alpha2  # [E, H]

        # Optional dropout on attention
        alpha = self.dropout(alpha)

        # MessagePassing: propagate v from source to target with weights alpha
        out = self.propagate(
            edge_index,
            x=v,         # v: [N, H, 2*Dh]
            alpha=alpha  # [E, H]
        )  # -> [N, H, 2*Dh] after aggregation

        # Merge heads
        out = out.reshape(N, self.embed_dim)  # [N, embed_dim]
        out = self.out_proj(out)

        # For logging, return alpha and beta
        return out, alpha, beta

    def message(self, x_j: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        x_j:   [E, H, 2*Dh]  (source node features for each edge)
        alpha: [E, H]        (attention coefficient per edge & head)
        """
        return alpha.unsqueeze(-1) * x_j  # [E, H, 2*Dh]


# ---------------------------------------------------------------------------
# ParticleAttentionBlock (sparse diff attention via edge_index/edge_attr)
# ---------------------------------------------------------------------------
class ParticleAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        u_embed: int,
        expansion_factor: float = 4.0,
        num_heads: int = 1,
        attn: str = "plain",
        num_layers: int = 1,
        layer_idx: int = 1,
        use_flash: bool = True,
        edge_in_dim: int = None,  # raw edge_attr dim (required for attn="diff")
    ):
        super(ParticleAttentionBlock, self).__init__()
        self.attn_type = attn
        self.embed_dim = embed_dim

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Attention module selection
        if attn in {"interaction", "plain"}:
            raise NotImplementedError("Dense attention is not implemented for sparse mode.")

        elif attn == "diff":
            # Sparse differential attention
            if edge_in_dim is None:
                raise ValueError(
                    "edge_in_dim must be provided for 'diff' attention to encode edge_attr."
                )

            # Edge encoder
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

        # MLP block
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
        x_res = x
        x = self.norm1(x)

        if self.attn_type == "diff":
            if edge_index is None or edge_attr is None:
                raise ValueError(
                    "For 'diff' attention, edge_index and edge_attr must be provided."
                )

            # Encode edge_attr once per block
            u_encoded = self.edge_encoder(edge_attr)  # [E, u_embed]

            attn_output, attn_weights, beta = self.pmha(
                x, edge_index, u_encoded, umask=None
            )
        else:
            # Dense attention path
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
        attn: str = "diff",
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
        self.u_embed_dim = 64  # encoded edge dim

        # Node embedding
        self.particle_embed = ParticleEmbedding(input_dim=in_channels)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            if attn == "diff":
                block = ParticleAttentionBlock(
                    embed_dim=embed_dim,
                    u_embed=self.u_embed_dim,
                    num_heads=num_heads,
                    attn="diff",
                    num_layers=num_blocks,
                    layer_idx=i,
                    use_flash=use_flash,
                    edge_in_dim=u_channels,
                )
            else:
                block = ParticleAttentionBlock(
                    embed_dim=embed_dim,
                    u_embed=self.u_embed_dim,
                    num_heads=num_heads,
                    attn=attn,
                    num_layers=num_blocks,
                    layer_idx=i,
                    use_flash=use_flash,
                )
            self.blocks.append(block)

        # Graph-level head
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
        x = data.x
        edge_index = getattr(data, "edge_index", None)
        edge_attr = getattr(data, "edge_attr", None)
        batch_idx = getattr(data, "batch", None)

        if batch_idx is None:
            batch_idx = x.new_zeros(x.size(0), dtype=torch.long)

        # 1) Node embedding
        x = self.particle_embed(x)  # [N, embed_dim]

        # 🔥 OPTIMIZATION: Don't collect attention/activations unless requested
        attn_weights_all = [] if return_attn_activations else None
        activations_all = [] if return_attn_activations else None
        beta_values = [] if self.attn == "diff" else None

        # 2) Stack attention blocks
        for block in self.blocks:
            if self.attn == "diff":
                x, attn, beta = block(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
                # 🔥 FIX: Only collect beta if needed, and detach immediately
                if beta_values is not None and beta is not None:
                    beta_val = beta.detach() if torch.is_tensor(beta) else beta
                    beta_values.append(beta_val)
                    
            elif self.attn == "plain":
                x, attn, _ = block(x)
            else:
                raise NotImplementedError(
                    "attn='interaction' with sparse edges is not implemented yet."
                )

            # 🔥 FIX: Detach immediately when collecting
            if return_attn_activations:
                attn_weights_all.append(attn.detach() if torch.is_tensor(attn) else attn)
                activations_all.append(x.detach().mean(dim=-1))  # [N]

        # 3) Graph pooling + classification head
        x_graph = global_mean_pool(x, batch_idx)  # [B, embed_dim]
        logits = self.mlp_head(x_graph)           # [B, num_classes]

        # 4) Mean beta (for diff attention logging)
        mean_beta = None
        if beta_values:
            # 🔥 FIX: Stack detached values
            beta_stack = torch.stack(beta_values)
            mean_beta = beta_stack.mean().item()  # Convert to Python scalar

        if return_attn_activations:
            attn_stack = torch.stack(attn_weights_all) if attn_weights_all else None
            act_stack = torch.stack(activations_all) if activations_all else None
            return logits, attn_stack, act_stack, mean_beta
        else:
            return logits, None, None, mean_beta


class JetTaggingModule(LightningModule):
    def __init__(
        self,
        in_channels: int = 7,
        u_channels: int = 6,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "diff",
        num_classes: int = 10,
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 1e-4,
        batch_size: int = 64,
        label_smoothing: float = 0.0,
        compile_model: bool = False,  # 🔥 CHANGED: Disable by default
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Build the model
        self.model = Transformer(
            in_channels=in_channels,
            u_channels=u_channels,
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            attn=attn,
            num_classes=num_classes,
        )

        self.compile_model = compile_model

        # Metrics for multiclass
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_auroc = AUROC(task="multiclass", num_classes=num_classes, average="macro")

        self.lr = lr
        self.eta_min = eta_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

        # 🔥 FIX: Only store predictions during explicit test phase
        self.test_predictions = []
        self.test_targets = []

    def _step(self, batch, prefix: str):
        """
        Optimized step with proper memory management.
        """
        y = batch.y
        
        # Forward pass
        logits, _, _, mean_beta = self.model(batch, return_attn_activations=False)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        # 🔥 FIX: Use separate metrics per phase to avoid reset issues
        if prefix == "train":
            acc_metric = self.train_acc
            auroc_metric = self.train_auroc
        elif prefix == "val":
            acc_metric = self.val_acc
            auroc_metric = self.val_auroc
        else:  # test
            acc_metric = self.test_acc
            auroc_metric = self.test_auroc

        # 🔥 FIX: Compute metrics in no_grad and detach everything
        with torch.no_grad():
            logits_detached = logits.detach()
            acc = acc_metric(logits_detached, y)
            probs = F.softmax(logits_detached, dim=-1)
            auc = auroc_metric(probs, y)

        sync_dist = self.trainer.num_devices > 1 if self.trainer else False
        actual_batch_size = y.size(0)
        is_train = prefix == "train"

        # 🔥 FIX: Log only scalars
        self.log(
            f"{prefix}_loss",
            loss.detach(),
            prog_bar=True,
            sync_dist=sync_dist,
            batch_size=actual_batch_size,
            on_step=is_train,  # Only log steps during training
            on_epoch=True
        )
        self.log(
            f"{prefix}_acc",
            acc,
            prog_bar=True,
            sync_dist=sync_dist,
            batch_size=actual_batch_size,
            on_step=False,
            on_epoch=True
        )
        self.log(
            f"{prefix}_auc",
            auc,
            prog_bar=False,
            sync_dist=sync_dist,
            batch_size=actual_batch_size,
            on_step=False,
            on_epoch=True
        )

        if mean_beta is not None:
            self.log(
                f"{prefix}_beta",
                mean_beta,  # Already a Python scalar
                prog_bar=False,
                sync_dist=sync_dist,
                batch_size=actual_batch_size,
                on_step=False,
                on_epoch=True
            )

        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self._step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        🔥 FIX: Minimal test step without attention collection.
        """
        with torch.no_grad():
            # Don't collect attention weights - too memory intensive
            logits, _, _, mean_beta = self.model(batch, return_attn_activations=False)
            y = batch.y
            
            loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)
            
            logits_detached = logits.detach()
            probs = F.softmax(logits_detached, dim=-1)
            
            acc = self.test_acc(logits_detached, y)
            auc = self.test_auroc(probs, y)

        sync_dist = self.trainer.num_devices > 1 if self.trainer else False

        self.log("test_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=self.batch_size)
        self.log("test_acc", acc, prog_bar=True, sync_dist=sync_dist, batch_size=self.batch_size, on_epoch=True)
        self.log("test_auc", auc, prog_bar=False, sync_dist=sync_dist, batch_size=self.batch_size, on_epoch=True)

        if mean_beta is not None:
            self.log("test_beta", mean_beta, prog_bar=False, sync_dist=sync_dist, batch_size=self.batch_size, on_epoch=True)

        # 🔥 FIX: Only store limited predictions
        if len(self.test_predictions) < 10000:  # Limit to prevent memory overflow
            self.test_predictions.append(probs.cpu().numpy())
            self.test_targets.append(y.cpu().numpy())

        return loss

    def on_train_epoch_end(self):
        """🔥 FIX: Reset training metrics."""
        self.train_acc.reset()
        self.train_auroc.reset()

    def on_validation_epoch_end(self):
        """🔥 FIX: Reset validation metrics."""
        self.val_acc.reset()
        self.val_auroc.reset()

    def on_test_epoch_end(self):
        """🔥 FIX: Clear accumulated predictions and reset metrics."""
        # Clear to free memory
        self.test_predictions.clear()
        self.test_targets.clear()
        self.test_acc.reset()
        self.test_auroc.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.t_max, eta_min=self.eta_min
        )
        return {
            "optimizer": opt,
            "lr_scheduler": sch,
        }