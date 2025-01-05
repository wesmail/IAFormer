# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lightning imports
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT

# Torch Metrics imports
from torchmetrics import Accuracy

# Framework imports
from models.transformer import mlp, Embedding, TransformerEncoder
from models.gnn import EEDGCNEncoder


class IAFormer(nn.Module):
    def __init__(
        self,
        f_dim,
        n_particles,
        U_features,
        k=7,
        n_Transformer=2,
        n_GNNLayers=4,
        h_dim=200,
        expansion_factor=4,
        n_heads=10,
        masked=True,
        pooling="avg",
        embed_dim=[128, 512, 128],
        U_dim=[128, 64, 64, 10],
        mlp_f_dim=[128, 64],
    ):
        super(IAFormer, self).__init__()

        """
        Args:
           f_dim: int, number  of the feature tokens
           n_particles: int, number  of the particle tokens
           U_features: int, number of the featires in the pairwise interaction matrix
           n_Transformer: int, number of Transformer layers
           h_dim: int, hidden dim of the Q,K and V
           expansion_factor: int, expansion of the size of the internal MLP layers in the Transformer layers.
           n_heads: int, number of attention heads
           masked: boolean, to use the attention mask
           Pooling: str, define the pooling kind, avg, max and sum
           embed_dim: list, define the number of neurons in the MLP for features embedding
           U_dim: list, define the number of neuron in the MLP for pairwise interaction embedding.
                                                                      The last number must equals the number of attention heads
           mlp_f_dim: list, define the number of neurons in the final MLP   
         
        return:
                transformer netwirk with pairwise interaction matrix included.
        """
        self.f_dim = f_dim
        self.n_particles = n_particles
        self.U_features = U_features
        self.k = k
        self.n_Transformer = n_Transformer
        self.n_GNNLayers = n_GNNLayers
        self.n_heads = n_heads
        self.masked = masked
        self.expansion_factor = expansion_factor
        self.h_dim = h_dim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.mlp_f_dim = mlp_f_dim
        self.U_dim = U_dim
        self.mlp = mlp(self.n_particles, self.mlp_f_dim)
        self.U_embeding = Embedding(self.U_features, self.U_dim)
        self.embed = Embedding(self.f_dim, self.embed_dim)
        self.encoder = TransformerEncoder(
            self.f_dim,
            embed_dim=self.embed_dim,
            h_dim=self.h_dim,
            num_layers=self.n_Transformer,
            expansion_factor=self.expansion_factor,
            n_heads=self.n_heads,
            masked=self.masked,
        )
        self.GNNencoder = EEDGCNEncoder(
            self.embed_dim[-1],
            self.embed_dim[-1],
            self.U_dim[-1],
            self.U_dim[-1],
            self.k,
            self.n_GNNLayers,
            self.pooling,
        )
        self.nW = nn.Linear(self.U_dim[-1], self.n_heads)
        self.nH = nn.LayerNorm(self.f_dim)
        self.nE = nn.LayerNorm(self.U_features)

    def forward(self, input_T, input_E):
        """
        input_T: dim (batch, particle tokens, feature tokens)
        input_E: dim (batch, particle tokens, particle tokens, pairwise features)
        """
        inp_E = self.U_embeding(input_E)
        inp_T = self.embed(input_T)
        out_H, out_E = self.GNNencoder(inp_T, inp_E)

        inp_E_T = torch.permute(self.nW(out_E), (0, -1, 1, 2))

        Transformer_out = self.encoder(input_T, inp_E_T)
        if self.pooling == "avg":
            Transformer_output = Transformer_out.mean(dim=2)
            out_H_ = out_H.mean(dim=2)

        elif self.pooling == "max":
            Transformer_output = Transformer_out.max(dim=2)[0]
            out_H_ = out_H.max(dim=2)[0]

        elif self.pooling == "sum":
            Transformer_output = Transformer_out.sum(dim=2)
            out_H_ = out_H.sum(dim=2)

        output_c = (
            Transformer_output + out_H_
        )  # torch.cat((Transformer_output,out_H_),dim=-1)
        output = self.mlp(output_c)

        return output


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
        D: int = 11,  # Number of node features
        N: int = 100,  # Number of particles in the event
        f: int = 4,  # Number of edge features
        n_Transformer: int = 8,  # Number of Transformer layers
        n_GNN: int = 3,  # Number of GNN layers
        k: int = 7,  # Number of neighbors or some specific constant
        expansion_factor: int = 4,  # Expansion factor of the internal MLP in the Transformer layers
        n_heads: int = 15,  # Number of attention heads
        masked: bool = True,  # If mask is used
        pooling: str = "avg",  # Pooling type, options are 'max', 'avg', or 'sum'
        embed_dim: list[int] = [256, 128, 64],  # Input embedding layers
        h_dim: int = 64,  # Hidden dimension of the scaling matrices
        U_dim: list[int] = [512, 256, 128, 64],  # Embedding layers of the edge matrix
        mlp_f_dim: list[int] = [512, 128, 64],  # Layers of the final MLP
        lr_step: int = 5,
        lr_gamma: float = 0.9,
    ) -> None:
        super().__init__()

        self.D = D
        self.N = N
        self.f = f
        self.n_Transformer = n_Transformer
        self.n_GNN = n_GNN
        self.k = k
        self.expansion_factor = expansion_factor
        self.n_heads = n_heads
        self.masked = masked
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.U_dim = U_dim
        self.mlp_f_dim = mlp_f_dim

        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        # self.save_hyperparameters()

        # Create an instance of the model
        self.model = IAFormer(
            self.D,
            self.N,
            self.f,
            self.n_Transformer,
            self.n_GNN,
            self.k,
            expansion_factor=self.expansion_factor,
            n_heads=self.n_heads,
            masked=self.masked,
            pooling=self.pooling,
            embed_dim=self.embed_dim,
            h_dim=self.h_dim,
            U_dim=self.U_dim,
            mlp_f_dim=self.mlp_f_dim,
        )

        # Accuarcy matric
        self.accuracy = Accuracy(task="binary")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single training step.

        Args:
            batch: The input batch of data.
            batch_idx: Index of the batch.

        Returns:
            STEP_OUTPUT: The training loss.
        """
        node_features, edge_features, masks, labels = (
            batch["node_features"],
            batch["edge_features"],
            batch["mask"],
            batch["labels"],
        )
        model_out = self.model(node_features, edge_features).flatten()
        # calculate the loss
        loss = torch.nn.functional.binary_cross_entropy(model_out, labels)
        acc = self.accuracy(model_out, labels)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("acc", acc, prog_bar=True, sync_dist=True)
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
        node_features, edge_features, masks, labels = (
            batch["node_features"],
            batch["edge_features"],
            batch["mask"],
            batch["labels"],
        )
        model_out = self.model(node_features, edge_features).flatten()
        # calculate the loss
        val_loss = torch.nn.functional.binary_cross_entropy(model_out, labels)
        val_acc = self.accuracy(model_out, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", val_acc, prog_bar=True, sync_dist=True)
        return val_loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """
        Perform a single test step.
        """
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_step)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step, gamma=self.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
