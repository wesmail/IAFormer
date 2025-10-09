# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ModuleList

# Lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# PyG
from torch_geometric.nn import (
    NNConv,
    TransformerConv,
    global_mean_pool,
    global_max_pool,
)


# Torch Metrics imports
from torchmetrics import Accuracy, AUROC


class ParticleEmbedding(torch.nn.Module):
    """
    Embeds particle features using an edge-conditioned convolution.
    """

    def __init__(self, in_channels=7, embed_dim=128, edge_dim=6, aggregation="mean"):
        """
        Initializes the ParticleEmbedding layer.

        Args:
            in_channels (int): Number of input particle features. Defaults to 7.
            embed_dim (int): Dimension of the embedded particle features. Defaults to 128.
            edge_dim (int): Dimension of the edge features. Defaults to 6.
            aggregation (str): Aggregation method for NNConv ('mean', 'sum', 'max'). Defaults to 'mean'.
        """
        super().__init__()

        self.edge_mlp = torch.nn.Sequential(
            nn.Linear(edge_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, in_channels * embed_dim),  # Weight shaping for NNConv
        )

        self.conv = NNConv(in_channels, embed_dim, self.edge_mlp, aggr=aggregation)

    def forward(self, x, edge_index, edge_attr):
        """
        Forward pass of the ParticleEmbedding layer.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, edge_dim].

        Returns:
            torch.Tensor: Embedded node features of shape [num_nodes, embed_dim].
        """
        return self.conv(x, edge_index, edge_attr)


class GNNWithTransformer(torch.nn.Module):
    """
    A Graph Neural Network (GNN) that combines Edge-Conditioned GCNs with Transformer Convolutions.
    """

    def __init__(
        self,
        in_channels=7,
        edge_dim=8,
        num_classes=1,
        embed_dims=[32, 64, 128],
        num_transformer_layers=6,
        transformer_heads=1,
        gnn_aggregation="mean",
        pooling_method="mean",
    ):
        """
        Initializes the GNNWithTransformer model.

        Args:
            in_channels (int): Number of input particle features. Defaults to 7.
            edge_dim (int): Dimension of the edge features. Defaults to 8.
            num_classes (int): Number of output classes. Defaults to 1.
            embed_dims (list[int]): A list of embedding dimensions for each GNN layer. Defaults to [32, 64, 128].
            num_transformer_layers (int): Number of Transformer Convolution layers. Defaults to 6.
            transformer_heads (int): Number of attention heads in the Transformer Convolution layers. Defaults to 1.
            gnn_aggregation (str): Aggregation method for the Edge-Conditioned GCN layers ('mean', 'sum', 'max'). Defaults to 'mean'.
            pooling_method (str): Global pooling method ('mean', 'max'). Defaults to 'mean'.
        """
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.num_gnn_layers = len(embed_dims)  # infer the number of layers
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = transformer_heads
        self.gnn_aggregation = gnn_aggregation
        self.pooling_method = pooling_method

        self.out_channels = embed_dims[-1]  # Output dimension of the GNN layers
        self.hidden_dim = 256
        # Dimension per head
        self.head_dim = self.hidden_dim // self.num_heads

        # Create Edge-Conditioned GCN Layers
        self.gnn_layers = nn.ModuleList(
            [
                ParticleEmbedding(
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    edge_dim=edge_dim,
                    aggregation=gnn_aggregation,
                )
                for i in range(self.num_gnn_layers)
            ]
        )

        self.proj = nn.Linear(self.out_channels, self.head_dim)

        # Create Transformer Convolution Layers
        self.transformer_layers = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=self.head_dim,
                    out_channels=self.head_dim,
                    heads=self.num_heads,
                    edge_dim=self.edge_dim,
                    beta=False,  # Enables learnable scaling factor
                    concat=False,  # DO NOT concatenates attention heads, average
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Create LayerNorm layers
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.head_dim) for _ in range(num_transformer_layers)]
        )

        # Define the MLP head for classification/regression
        self.mlp_head = nn.Linear(self.head_dim, num_classes)

    def forward(self, x, edge_index, edge_attr, batch=None):
        """
        Forward pass of the GNNWithTransformer model.

        Args:
            x (torch.Tensor): Node features of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format [2, num_edges].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, edge_dim].
            batch (torch.Tensor, optional): Batch vector assigning each node to a specific graph. Defaults to None.

        Returns:
            torch.Tensor: Output of the MLP head, typically representing class scores or regression values.
        """

        # Apply Edge-Conditioned GCN layers
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, edge_index, edge_attr)

        # Expand dim to hidden_dim
        x = self.proj(x)

        # Apply Transformer Convolution layers with skip connections and LayerNorm
        for i in range(self.num_transformer_layers):
            x_residual = x  # Store the input for the skip connection
            x = self.transformer_layers[i](x, edge_index, edge_attr)
            x = self.norms[i](x + x_residual)  # Skip connection followed by LayerNorm
            x = F.gelu(x)  # Apply GELU activation

        # Global pooling
        if self.pooling_method == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_method == "max":
            x = global_max_pool(x, batch)
        else:
            raise ValueError(
                f"Unsupported pooling method: {self.pooling_method}.  Choose 'mean' or 'max'."
            )

        # Apply the MLP head
        x = self.mlp_head(x)
        return x


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
        edge_dim: int = 8,
        num_classes: int = 1,
        embed_dims: list = [32, 64, 128],
        num_transformer_layers: int = 6,
        transformer_heads: int = 1,
        gnn_aggregation: str = "mean",
        pooling_method: str = "mean",
        eta_min: float = 1e-5,
        t_max: int = 20,
        lr: float = 0.0001,
        batch_size: int = 64,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.edge_dim = edge_dim
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_transformer_layers = num_transformer_layers
        self.transformer_heads = transformer_heads
        self.gnn_aggregation = gnn_aggregation
        self.pooling_method = pooling_method
        self.lr = lr
        self.eta_min = eta_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.model = GNNWithTransformer(
            in_channels=self.in_channels,
            embed_dims=self.embed_dims,
            edge_dim=self.edge_dim,
            num_transformer_layers=self.num_transformer_layers,
            gnn_aggregation=self.gnn_aggregation,
            pooling_method=self.pooling_method,
            num_classes=self.num_classes,
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
        model_out = self.model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        ).flatten()
        # calculate the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model_out, batch.y)
        acc = self.accuracy(model_out, batch.y)
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
        model_out = self.model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        ).flatten()
        # calculate the loss
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model_out, batch.y
        )
        val_acc = self.accuracy(model_out, batch.y)
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
        model_out = self.model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        ).flatten()
        # calculate the loss
        test_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model_out, batch.y
        )
        self.log(
            "test_loss",
            test_loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_acc",
            self.accuracy(model_out, batch.y),
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "test_auc",
            self.aucroc(model_out, batch.y),
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
