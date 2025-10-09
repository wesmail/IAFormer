# Generic imports
import math

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# PyG
# from torch_geometric.nn import MessagePassing, knn_graph, global_mean_pool

# Torch Metrics imports
from torchmetrics import Accuracy, AUROC

# Framework imports
from models.particle_net import ParticleNet
from models.particle_transformer import InteractionInputEncoding, ParticleAttentionBlock


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        in_channels=7,
        u_channels=6,
        num_pnet_layers=3,
        pnet_embed_dims=[32, 64, 128],
        knn=[4, 8, 16],
        embed_dim=128,
        num_heads=8,
        num_blocks=6,
        num_classes=1,
    ):
        super(ParticleTransformer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.u_embed_dim = num_heads * 2
        self.particle_embed = ParticleNet(
            in_channels=in_channels,
            num_layers=num_pnet_layers,
            embed_dims=pnet_embed_dims,
            k=knn,
        )
        self.interaction_embed = InteractionInputEncoding(
            input_dim=u_channels, output_dim=self.u_embed_dim
        )

        self.blocks = nn.ModuleList(
            [
                ParticleAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_layers=num_blocks,
                    layer_idx=i,
                )
                for i in range(num_blocks)
            ]
        )

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, pos, u=None, pmask=None):
        # particles mask
        if pmask is not None:
            pad_mask = (pmask == 1).sum(
                dim=-1
            )  # set to -1 (pad_mask = -1) to pool across features
            x = x[:, : pad_mask.max()]
            pos = pos[:, : pad_mask.max()]
            u = self.interaction_embed(u[:, : pad_mask.max(), : pad_mask.max(), :])
        x = self.particle_embed(x, pos)
        # adjcancy matrix mask
        umask = (
            (u == 0)[:, : pad_mask.max(), : pad_mask.max(), 0]
            .bool()
            .unsqueeze(-1)
            .repeat(1, 1, 1, self.u_embed_dim)
        )

        for block in self.blocks:
            x = block(x, u, umask)

        # Aggregate features (e.g., mean pooling)
        # TODO: let user choose pooling function
        x = x.mean(
            dim=1
        )  # Pool across particles (dim=1) features (dim=-1) MUST use pad_mask = -1

        logits = self.mlp_head(x)
        return logits


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
        num_pnet_layers: int = 3,
        pnet_embed_dims: list = [32, 64, 128],
        knn: list = [4, 8, 16],
        num_blocks: int = 8,
        num_heads: int = 8,
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 0.0001,
        batch_size: int = 64,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.eta_min = eta_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.model = ParticleTransformer(
            in_channels=in_channels,
            u_channels=u_channels,
            num_pnet_layers=num_pnet_layers,
            pnet_embed_dims=pnet_embed_dims,
            knn=knn,
            embed_dim=pnet_embed_dims[-1],
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_classes=1,
        )

        # Accuarcy matric
        self.accuracy = Accuracy(task="binary")
        self.aucroc = AUROC(task="binary")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single training step.

        Args:
            batch: The input batch of data.
            batch_idx: Index of the batch.

        Returns:
            STEP_OUTPUT: The training loss.
        """
        x, pos, u, y, pmask = (
            batch["node_features"],
            batch["coordinates"],
            batch["edge_features"],
            batch["labels"],
            batch["node_mask"],
        )
        model_out = self.model(x, pos, u, pmask).flatten()
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
        x, pos, u, y, pmask = (
            batch["node_features"],
            batch["coordinates"],
            batch["edge_features"],
            batch["labels"],
            batch["node_mask"],
        )
        model_out = self.model(x, pos, u, pmask).flatten()
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
        x, pos, u, y, pmask = (
            batch["node_features"],
            batch["coordinates"],
            batch["edge_features"],
            batch["labels"],
            batch["node_mask"],
        )
        model_out = self.model(x, pos, u, pmask).flatten()
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
