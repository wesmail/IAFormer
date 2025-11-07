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
        num_classes=10,
        use_flash=True,  # NEW: Pass flash attention flag
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
                use_flash=use_flash,  # Pass to attention blocks
            )
            for i in range(num_blocks)
        ])

        self.mlp_head = nn.Linear(max_num_particles, num_classes)

    def forward(self, x, u=None, umask=None, return_attn_activations=False):
        """
        Args:
            return_attn_activations: If True, return attention weights and activations (used for test/eval only)
        """
        x = self.particle_embed(x)

        if u is not None:
            # OPTIMIZED: Create mask properly
            # u shape: [B, P, P, F] where F is feature dim
            u_mask_bool = (u != 0)[..., 0].bool()  # [B, P, P]
            # Add feature dimension and expand to match head_dim
            batch_size = u_mask_bool.shape[0]
            num_particles = u_mask_bool.shape[1]
            #umask = u_mask_bool.unsqueeze(-1).expand(batch_size, num_particles, num_particles, self.head_dim)
            u = self.interaction_embed(u)

        # Only collect attention/activations if needed (for test/eval)
        if return_attn_activations:
            attn_weights = []
            activations = []
        
        # Collect beta values for logging (only for diff attention)
        beta_values = [] if self.attn == "diff" else None

        for block in self.blocks:
            if self.attn == "plain":
                x, attn, _ = block(x)
            else:
                x, attn, beta = block(x, u)
                if self.attn == "diff" and beta is not None:
                    beta_values.append(beta)

            if return_attn_activations:
                attn_weights.append(attn)
                activations.append(x.mean(dim=-1))

        x = x.mean(dim=-1)
        logits = self.mlp_head(x)
        
        # Compute mean beta across all blocks for logging
        mean_beta = None
        if beta_values:
            beta_stack = torch.stack(beta_values)
            mean_beta = beta_stack.mean().item()
        
        if return_attn_activations:
            attn_stack = torch.stack(attn_weights).transpose(1, 0)
            act_stack  = torch.stack(activations).transpose(1, 0)
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
        attn: str = "plain",
        max_num_particles: int = 100,
        num_classes: int = 10,
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 1e-4,
        batch_size: int = 64,
        label_smoothing: float = 0.0,
        compile_model: bool = True,  # NEW: torch.compile flag
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
            max_num_particles=max_num_particles,
            attn=attn,
            num_classes=num_classes,
        )

        # CRITICAL: Compile AFTER model construction but BEFORE any forward passes
        # This is done in configure_model() which Lightning calls at the right time
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

    def configure_model(self):
        """
        BEST PLACE TO COMPILE: Lightning calls this hook at the right time
        (after model is moved to device, before training starts)
        """
        if self.compile_model and not hasattr(self, '_compiled'):
            print("Compiling model with torch.compile...")
            
            # OPTION 1: Compile the entire model (recommended for most cases)
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",  # Options: "default", "reduce-overhead", "max-autotune"
                fullgraph=False,  # Set True only if no dynamic control flow
            )
            
            # OPTION 2: Compile only the forward pass (alternative)
            # self.model.forward = torch.compile(
            #     self.model.forward,
            #     mode="reduce-overhead",
            #     fullgraph=False,
            # )
            
            self._compiled = True
            print("✅ Model compilation complete!")

    # ---- helpers ----
    def _step(self, batch, prefix: str):
        x, u, y = batch["node_features"], batch["edge_features"], batch["labels"]
        logits, _, _, mean_beta = self.model(x, u, return_attn_activations=False)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        # Metrics
        acc = self.acc(logits, y)
        probs = logits.softmax(dim=-1)
        auc = self.auroc(probs, y)

        # Optimize logging: sync_dist only when needed
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False

        # Reduce logging overhead
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=self.batch_size)
        self.log(f"{prefix}_acc",  acc,  prog_bar=False, sync_dist=sync_dist, batch_size=self.batch_size, on_step=False, on_epoch=True)
        self.log(f"{prefix}_auc",  auc,  prog_bar=False, sync_dist=sync_dist, batch_size=self.batch_size, on_step=False, on_epoch=True)

        # Log beta value if available (for diff attention)
        if mean_beta is not None:
            self.log(f"{prefix}_beta", mean_beta, prog_bar=False, sync_dist=sync_dist, batch_size=self.batch_size, on_step=False, on_epoch=True)

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
        # For test, we need attention weights and activations
        logits, attn_weights, activations, mean_beta = self.model(x, u, return_attn_activations=True)
        loss = F.cross_entropy(logits, y, label_smoothing=self.label_smoothing)

        probs = logits.softmax(dim=-1)
        sync_dist = self.trainer.num_devices > 1 if self.trainer else False
        self.log("test_loss", loss, prog_bar=True, sync_dist=sync_dist, batch_size=self.batch_size)
        self.log("test_acc",  self.acc(logits, y), prog_bar=True, sync_dist=sync_dist, batch_size=self.batch_size)
        self.log("test_auc",  self.auroc(probs, y), prog_bar=True, sync_dist=sync_dist, batch_size=self.batch_size)

        # Log beta value if available
        if mean_beta is not None:
            self.log("test_beta", mean_beta, prog_bar=False, sync_dist=sync_dist, batch_size=self.batch_size)

        # stash (optional)
        self.test_predictions.extend(probs.detach().cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        if attn_weights is not None:
            self.attn_weights.extend(attn_weights.detach().cpu().numpy())
        if activations is not None:
            self.activations.extend(activations.detach().cpu().numpy())

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.t_max, eta_min=self.eta_min)
        return {"optimizer": opt, "lr_scheduler": sch}
