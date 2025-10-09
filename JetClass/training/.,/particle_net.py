import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicEdgeConv(nn.Module):
    def __init__(self, in_channels, embed_dim, k):
        super(DynamicEdgeConv, self).__init__()
        self.k = k

        # MLP for feature transformation
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, embed_dim, bias=True),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Shortcut connection (identity if dimensions match, otherwise use a Linear layer)
        if in_channels == embed_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(in_channels, embed_dim)

    def forward(self, x, coordinates):
        """
        Args:
            x: Input features [B, N, D]
            coordinates: Point cloud coordinates [B, N, 3] (or any spatial dim)
        Returns:
            x_out: Updated features after EdgeConv, shape [B, N, embed_dim]
        """
        B, N, D = x.shape

        # Step 1: Compute pairwise distances and get k-nearest neighbors
        pairwise_dist = torch.cdist(coordinates, coordinates, p=2)  # [B, N, N]
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
        )  # [B * N * k, embed_dim]
        combined_features = combined_features.view(
            B, N, self.k, -1
        )  # [B, N, k, embed_dim]

        # Aggregate (max pooling across neighbors)
        aggregated_features = combined_features.max(dim=2)[0]  # [B, N, embed_dim]

        # Step 5: Apply shortcut connection
        shortcut = self.shortcut(x)  # Apply Linear layer if needed
        x_out = aggregated_features + shortcut  # Add shortcut connection

        return F.gelu(x_out)

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

    def forward(self, x, pos):
        # Pass input through each DynamicEdgeConv layer
        for conv in self.edge_conv:
            x = conv(x, pos)

        return x

