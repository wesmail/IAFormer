# Generic imports
import math

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# Framework imports
from models.embedding import ParticleEmbedding, InteractionInputEncoding
from models.attention import ParticleAttentionBlock

# Torch Metrics imports
from torchmetrics import Accuracy, AUROC


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
        num_classes=1,
    ):
        super(Transformer, self).__init__()
        if attn not in {"plain", "diff", "interaction"}:
            raise ValueError(
                f"Invalid attention type: {attn}. Must be 'plain', 'interaction' or 'diff'."
            )
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.u_embed_dim = 64
        self.attn = attn
        self.head_dim = (
            num_heads
            if attn in {"plain", "interaction"}
            else (num_heads * 2 if attn == "diff" else -1)
        )
        print(f"Head dim is {self.head_dim}\n")
        self.particle_embed = ParticleEmbedding(input_dim=in_channels)
        self.interaction_embed = InteractionInputEncoding(
            input_dim=u_channels,
            output_dim=self.u_embed_dim,
        )

        self.blocks = nn.ModuleList(
            [
                ParticleAttentionBlock(
                    embed_dim=embed_dim,
                    u_embed=self.u_embed_dim,
                    num_heads=num_heads,
                    attn=attn,
                    num_layers=num_blocks,
                    layer_idx=i,
                )
                for i in range(num_blocks)
            ]
        )

        self.mlp_head = nn.Linear(max_num_particles, num_classes)

    def forward(self, x, u=None, umask=None):
        x = self.particle_embed(x)
        # adjcancy matrix mask
        if u is not None:
            umask = (u != 0)[..., 0].bool().unsqueeze(-1).repeat(1, 1, 1, self.head_dim)
            u = self.interaction_embed(u)
        attn_weights = []
        activations = []
        for block in self.blocks:
            if self.attn == "plain":
                x, attn = block(x)
            elif self.attn in {"interaction", "diff"}:
                x, attn = block(x, u)
            
            attn_weights.append(attn)
            activations.append(x.mean(dim=-1)) # output shape [batch size, number of particles]

        # Aggregate features (e.g., mean pooling)
        # TODO: let user choose pooling function
        x = x.mean(dim=-1)  # Pool across particles (dim=1) features (dim=-1)

        logits = self.mlp_head(x)
        return logits, torch.stack(attn_weights).transpose(1, 0), torch.stack(activations).transpose(1, 0)


class JetTaggingModule(LightningModule):
    """
    PyTorch Lightning Module for training a jet tagging model using EdgeConv.

    Args:
        in_channels (int): Number of input channels for the model.
        out_channels_n (int): Number of output channels for the node features.
        out_channels_E (int): Number of output channels for the edge features.
        k (int): Number of nearest neighbors for the k-NN graph.
        lr_step (int): Step size for the learning rate scheduler.
        lr_gamma (float): Multiplicative factor of learning rate decay.
    """

    def __init__(
        self,
        in_channels: int = 7,
        u_channels: int = 6,
        embed_dim: int = 32,
        num_heads: int = 8,
        num_blocks: int = 6,
        attn: str = "plain",
        max_num_particles: int = 100,
        num_classes: int = 1,
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 0.0001,
        batch_size: int = 64,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.u_channels = u_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.attn = attn
        self.max_num_particles = max_num_particles
        self.num_classes = num_classes

        self.lr = lr
        self.eta_min = eta_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.model = Transformer(
            in_channels=self.in_channels,
            u_channels=self.u_channels,
            embed_dim=self.embed_dim,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            max_num_particles=self.max_num_particles,
            attn=self.attn,
            num_classes=self.num_classes,
        )

        # Accuarcy matric
        self.accuracy = Accuracy(task="binary")
        self.aucroc = AUROC(task="binary")

        # for predictions
        self.test_predictions = []
        self.test_targets = []
        self.attn_weights = []
        self.activations = []
        

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single training step.

        Args:
            batch: The input batch of data.
            batch_idx: Index of the batch.

        Returns:
            STEP_OUTPUT: The training loss.
        """
        x, u, y = (
            batch["node_features"],
            batch["edge_features"],
            batch["labels"],
        )
        model_out, _, _ = self.model(x, u)
        model_out = model_out.flatten()
        # calculate the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model_out, y)
        acc = self.accuracy(model_out, y)
        self.log(
            "loss", loss, prog_bar=True, sync_dist=True, batch_size=self.batch_size
        )
        self.log("acc", acc, prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single validation step.

        Args:
            batch: The input batch of data.
            batch_idx: Index of the batch.

        Returns:
            STEP_OUTPUT: The validation loss.
        """
        x, u, y = (
            batch["node_features"],
            batch["edge_features"],
            batch["labels"],
        )
        model_out, _, _ = self.model(x, u)
        model_out = model_out.flatten()
        # calculate the loss
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(model_out, y)
        val_acc = self.accuracy(model_out, y)
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_acc",
            val_acc,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        return val_loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single test step.
        """
        x, u, y = (
            batch["node_features"],
            batch["edge_features"],
            batch["labels"],
        )
        model_out, attn_weights, activations = self.model(x, u)
        model_out = model_out.flatten()
        # calculate the loss
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(model_out, y)
        self.log(
            "test_loss",
            test_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_acc",
            self.accuracy(model_out, y),
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_auc",
            self.aucroc(model_out, y),
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # for predictions
        self.test_predictions.extend(model_out.detach().cpu().numpy())
        self.test_targets.extend(y.cpu().numpy())
        self.attn_weights.extend(attn_weights.cpu().numpy())
        self.activations.extend(activations.cpu().numpy())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}