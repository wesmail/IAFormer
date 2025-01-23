# Generic imports
import math

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
from models.transformer import Transformer_P


# ******************************************************************************************************************************************
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class ParticleEmbedding(nn.Module):
    def __init__(self, input_dim=11):
        super(ParticleEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


class InteractionInputEncoding(nn.Module):
    def __init__(self, input_dim=4, output_dim=8):
        super(InteractionInputEncoding, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        bs, tokens = x.size(0), x.size(1)
        x = x.view(bs, -1, x.size(-1)).permute(0, 2, 1)
        x = self.conv(x).permute(0, 2, 1)

        return x.view(bs, tokens, tokens, x.size(-1))


class MultiheadDiffAttn(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Dimension per head
        self.head_dim = embed_dim // num_heads // 2  # 2 attention maps
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert embed_dim % 2 == 0, "embed_dim must be divisible by 2"
        self.scaling = self.head_dim**-0.5

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Lambda parameters
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k1 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_q2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )
        self.lambda_k2 = nn.Parameter(
            torch.zeros(self.head_dim).normal_(mean=0, std=0.1)
        )

        # Normalization layer
        self.subln = nn.LayerNorm(self.head_dim * 2)

    def forward(self, x, u=None):
        batch_size, num_tokens, _ = x.size()

        # Linear projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape into heads
        q = q.view(batch_size, num_tokens, self.num_heads * 2, self.head_dim).transpose(
            1, 2
        )  # shape (bs, num_heads*2, num_tokens, head_dim)
        k = k.view(batch_size, num_tokens, self.num_heads * 2, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, num_tokens, self.num_heads, 2 * self.head_dim).transpose(
            1, 2
        )

        # Scale queries
        q *= self.scaling

        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        if u is not None:
            mask = ~u.bool()
            u.masked_fill_(mask, -torch.inf)
            attn_weights += u.transpose(3, 1)

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute lambda
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float())
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Apply differential attention
        attn_weights = attn_weights.view(
            batch_size, self.num_heads, 2, num_tokens, num_tokens
        )
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        # Weighted sum
        attn = torch.matmul(attn_weights, v)

        # Normalize and reshape
        attn = self.subln(attn)
        attn = attn.transpose(1, 2).reshape(batch_size, num_tokens, self.embed_dim)

        # Final projection
        attn = self.out_proj(attn)
        return attn


class ParticleAttentionBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, num_heads=1, num_layers=1):
        super(ParticleAttentionBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pmha = MultiheadDiffAttn(
            embed_dim=embed_dim, num_heads=num_heads, depth=num_layers
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.GELU(),
            nn.LayerNorm(expansion_factor * embed_dim),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )

    def forward(self, x, u=None):
        x_res = x
        x = self.norm1(x)
        attn_output = self.pmha(x, u)  # x, and u embeddings
        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h


class DynamicEdgeConv(nn.Module):
    def __init__(self, in_channels, embed_dim, k, out_channels=None):
        super(DynamicEdgeConv, self).__init__()
        self.k = k
        out_channels = embed_dim if out_channels is None else out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        """
        Args:
            x: Input point cloud data, shape [B, N, D]
               B - batch size, N - number of points, D - feature dimensions
        Returns:
            x_out: Updated features after EdgeConv, shape [B, N, out_channels]
        """
        B, N, D = x.size()

        # Step 1: Compute pairwise distance and get k-nearest neighbors
        # TODO: remove hard-coded 8 and 9 to replace with eta and phi
        pairwise_dist = torch.cdist(x[..., [8, 9]], x[..., [8, 9]], p=2)  # [B, N, N]
        idx = pairwise_dist.topk(k=self.k, dim=-1, largest=False)[1]  # [B, N, k]

        # Step 2: Gather neighbor features
        neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, N, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, D),
        )  # [B, N, k, D]

        # Central point repeated for k neighbors: [B, N, k, D]
        central = x.unsqueeze(2).expand(-1, -1, self.k, -1)

        # Step 3: Compute edge features
        relative_features = neighbors - central  # [B, N, k, D]
        combined_features = torch.cat(
            [central, relative_features], dim=-1
        )  # [B, N, k, 2*D]

        # Step 4: Apply MLP and aggregation
        combined_features = self.mlp(
            combined_features.view(-1, 2 * D)
        )  # [B * N * k, out_channels]
        combined_features = combined_features.view(
            B, N, self.k, -1
        )  # Reshape to [B, N, k, out_channels]

        # Aggregate (avg pooling across neighbors)
        x_out = combined_features.mean(dim=2)  # [B, N, out_channels]

        return x_out


class ParticleNet(nn.Module):
    def __init__(
        self, in_channels=11, num_layers=3, embed_dims=[64, 128, 256], k=[16, 16, 16]
    ):
        super(ParticleNet, self).__init__()

        # Ensure embed_dims and k are lists
        assert isinstance(
            embed_dims, list
        ), f"Expected embed_dims to be a list, but got {type(embed_dims)}"
        assert isinstance(k, list), f"Expected k to be a list, but got {type(k)}"

        # Assertion to ensure embed_dims and k have length 3
        assert (
            len(embed_dims) == num_layers
        ), f"Expected embed_dims to have the same length as 'num_layers={num_layers}', but got {len(embed_dims)}"
        assert (
            len(k) == num_layers
        ), f"Expected k to have length the same length as 'num_layers={num_layers}', but got {len(k)}"

        # Creating a list of DynamicEdgeConv layers
        self.edge_conv = nn.ModuleList(
            [
                DynamicEdgeConv(
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    k=k[i],
                )
                for i in range(num_layers)
            ]
        )

        # self.classifier = nn.Sequential(nn.Linear(embed_dims[-1], embed_dims[-1]),
        #                                nn.GELU(),
        #                                nn.Dropout(0.1),
        #                                nn.Linear(embed_dims[-1], 1))

    def forward(self, x):
        # Pass input through each DynamicEdgeConv layer
        for conv in self.edge_conv:
            x = conv(x)

        # Aggregate over the token dim
        # x_out = x.mean(dim=1)

        return x


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        feat_particles_dim,
        feat_interaction_dim,
        embed_dim,
        num_heads,
        num_blocks,
        k=[16, 32, 32],
        num_classes=1,
    ):
        super(ParticleTransformer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.particle_embed = ParticleNet(in_channels=feat_particles_dim, k=k)
        self.interaction_embed = InteractionInputEncoding(
            input_dim=feat_interaction_dim, output_dim=num_heads * 2
        )
        self.blocks = nn.ModuleList(
            [
                ParticleAttentionBlock(
                    embed_dim=embed_dim, num_heads=num_heads, num_layers=num_blocks
                )
                for _ in range(num_blocks)
            ]
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, particles, interactions):
        x = self.particle_embed(particles)
        u = self.interaction_embed(interactions)

        for block in self.blocks:
            x = block(x, u)

        # Aggregate features (e.g., mean pooling)
        # TODO: let user choose pooling function
        x = x.mean(dim=1)  # Pool across particles

        logits = self.mlp_head(x)
        return logits


# ******************************************************************************************************************************************


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
        lr: float = 0.0001,
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

        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        # self.save_hyperparameters()

        self.model = ParticleTransformer(
            feat_particles_dim=11,
            feat_interaction_dim=4,
            embed_dim=256,  # TODO: Must match last embed dim for particle-net
            num_heads=8,
            num_blocks=4,
            k=[16, 16, 32],
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
        x, edge_attr, labels = (
            batch["node_features"],
            batch["edge_features"],
            batch["labels"],
        )
        model_out = self.model(x, edge_attr).flatten()
        # calculate the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model_out, labels)
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
        x, edge_attr, labels = (
            batch["node_features"],
            batch["edge_features"],
            batch["labels"],
        )
        model_out = self.model(x, edge_attr).flatten()
        # calculate the loss
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model_out, labels
        )
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.lr_step, gamma=self.lr_gamma
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
