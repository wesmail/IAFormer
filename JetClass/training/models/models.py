# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# Torch Metrics imports
from torchmetrics import Accuracy, AUROC

# Your modules
from models.embedding import ParticleEmbedding, InteractionInputEncoding
from models.attention import ParticleAttentionBlock


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels=7,
        u_channels=6,
        embed_dim=32,
        num_heads=8,
        num_blocks=6,
        attn="plain",
        max_num_particles=100,
        num_classes=10,              # <-- multiclass
    ):
        super().__init__()
        if attn not in {"plain", "diff", "interaction"}:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be 'plain', 'interaction' or 'diff'."
            )
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.u_embed_dim = 64
        self.attn = attn
        self.head_dim = num_heads if attn in {"plain", "interaction"} else (num_heads * 2)

        self.particle_embed = ParticleEmbedding(input_dim=in_channels)
        self.interaction_embed = InteractionInputEncoding(
            input_dim=u_channels,
            output_dim=self.u_embed_dim,
        )

        self.blocks = nn.ModuleList([
            ParticleAttentionBlock(
                embed_dim=embed_dim,
                u_embed=self.u_embed_dim,
                num_heads=num_heads,
                attn=attn,
                num_layers=num_blocks,
                layer_idx=i,
            )
            for i in range(num_blocks)
        ])

        # A learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # NOTE:
        # You pool with x.mean(dim=-1) -> shape [B, P].
        # So a Linear(P -> num_classes) is appropriate without changing your tensor layout.
        self.mlp_head = nn.Linear(max_num_particles, num_classes)

    def forward(self, x, u=None, umask=None):
        x = self.particle_embed(x)  # expect shape [B, P, E] downstream
        # --- Add CLS token ---
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape: [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)  # shape: [B, 1 + P, d_model]

        if u is not None:
            # adjacency matrix mask (kept same semantics)
            umask = (u != 0)[..., 0].bool().unsqueeze(-1).repeat(1, 1, 1, self.head_dim)
            u = self.interaction_embed(u)

        attn_weights = []
        activations = []
        for block in self.blocks:
            if self.attn == "plain":
                x, attn, _ = block(x)
            else:  # "interaction" or "diff"
                x, attn, beta = block(x, u)

            attn_weights.append(attn)
            # mean over feature dim → [B, P]
            activations.append(x.mean(dim=-1))

        # --- Pool across tokens using mean ---
        x = x[:, 0]  # Take only the CLS token output
        # Your current pooling: mean over feature dim (last dim) → [B, P]
        #x = x.mean(dim=-1)

        logits = self.mlp_head(x)  # [B, num_classes]
        # stack lists to tensors for logging/analysis
        attn_stack = torch.stack(attn_weights).transpose(1, 0)
        act_stack  = torch.stack(activations).transpose(1, 0)
        return logits, attn_stack, act_stack


class JetTaggingModule(LightningModule):
    def __init__(
        self,
        in_channels: int = 7,
        u_channels: int = 6,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "plain",
        max_num_particles: int = 100,
        num_classes: int = 10,          # <-- multiclass
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 1e-4,
        batch_size: int = 64,
        label_smoothing: float = 0.0,   # optional
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

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

    # ---- helpers ----
    def _step(self, batch, prefix: str):
        x, u, y = batch["node_features"], batch["edge_features"], batch["labels"]  # y: [B] long in [0..9]
        logits, _, _ = self.model(x, u)  # [B, C]
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        # Metrics
        acc = self.acc(logits, y)  # torchmetrics will argmax internally for multiclass
        # For AUROC-multiclass we need probabilities:
        probs = logits.softmax(dim=-1)
        auc = self.auroc(probs, y)

        self.log(f"{prefix}/loss", loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f"{prefix}/acc",  acc,  prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log(f"{prefix}/auc",  auc,  prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss, logits

    # ---- Lightning hooks ----
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "train")
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, _ = self._step(batch, "val")
        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, u, y = batch["node_features"], batch["edge_features"], batch["labels"]
        logits, attn_weights, activations = self.model(x, u)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        probs = logits.softmax(dim=-1)
        self.log("test/loss", loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/acc",  self.acc(logits, y), prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/auc",  self.auroc(probs, y), prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # stash (optional)
        self.test_predictions.extend(probs.detach().cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        self.attn_weights.extend(attn_weights.detach().cpu().numpy())
        self.activations.extend(activations.detach().cpu().numpy())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=self.eta_min)
        return {"optimizer": opt, "lr_scheduler": sch}

