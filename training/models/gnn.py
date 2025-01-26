import torch
import torch.nn as nn

class EEDGCNEncoder(nn.Module):
    def __init__(self,in_channels_n,  out_channels_n,in_channels_E,out_channels_E, k,n_layers, pooling='avg'):
        super(EEDGCNEncoder,self).__init__()
        self.in_channels_n=in_channels_n
        self.out_channels_n=out_channels_n
        self.out_channels_E=out_channels_E
        self.in_channels_E=in_channels_E
        self.k=k
        self.n_layers=n_layers
        self.pooling=pooling
        self.layers = nn.ModuleList([EdgeConvWithEdgeFeatures(self.in_channels_n, self.out_channels_n,self.out_channels_E,self.k,self.pooling) for i in range(self.n_layers)])
        self.nH = nn.LayerNorm(self.in_channels_n)
        self.nE = nn.LayerNorm(self.in_channels_E)
    def forward(self,H,E):

        out_H = self.nH(H)
        out_E = self.nE(E)

        for layer in self.layers:
            out_H,out_E = layer(out_H,out_E)

        return out_H,out_E

class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.2):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, node,node, dim_e]
        :param node: [batch, node, node, dim]
        :return:
        """

        node = torch.cat([node1, node2], dim=-1) # [batch, node, node, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, node,npde, dim_e]

class EdgeConvWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels,  out_channels_n,out_channels_E, k, pooling='avg'):
        super(EdgeConvWithEdgeFeatures, self).__init__()
        self.k = k
        self.pooling=pooling
        self.in_channels = in_channels
        self.out_channels_n = out_channels_n
        self.out_channels_E = out_channels_E
        self.W = nn.Linear(self.in_channels, self.out_channels_E)
        self.highway = Edgeupdate(self.in_channels, self.out_channels_E, dropout_ratio=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.in_channels, self.out_channels_n, bias=False),
            nn.BatchNorm1d(self.out_channels_n),
            nn.GELU()
        )
        
    def forward(self, x,weight_adj):
        """
        Args:
            x: Input point cloud data, shape [B, N, D]
               B - batch size, N - number of points, D - feature dimensions
            edge_features: Input edge features, shape [B, N, k, E]
               E - edge feature dimensions
        Returns:
            x_out: Updated features after EdgeConv, shape [B, N, out_channels]
        """
        B, N, D = x.size()
        _, _, _, E = weight_adj.size()
        
        # Step 1: Compute pairwise distance and get k-nearest neighbors
        pairwise_dist = torch.cdist(x, x, p=2)  # [B, N, N]
        idx = pairwise_dist.topk(k=self.k, dim=-1, largest=False)[1]  # [B, N, k]
        
        # Step 2: Gather neighbor features
        neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, N, -1), 
            2, 
            idx.unsqueeze(-1).expand(-1, -1, -1, D)
        )  # [B, N, k, D]
        
        # Central point repeated for k neighbors: [B, N, k, D]
        central = x.unsqueeze(2).expand(-1, -1, self.k, -1)
        
        # Step 3: Compute edge features
        relative_features = neighbors - central  # [B, N, k, D]
        combined_features = torch.cat([central, relative_features], dim=-1)  # [B, N, k, 2*D + E]
        
        # Step 4: Apply MLP and aggregation
        combined_features = self.mlp(combined_features.view(-1, 2 * D))  # [B * N * k, out_channels]
        combined_features = combined_features.view(B, N, self.k, -1)  # Reshape to [B, N, k, out_channels]
        
        if self.pooling == 'avg':
            n_out = combined_features.mean(dim=2)
        elif self.pooling == 'max':
            n_out = combined_features.max(dim=2)[0]
        elif self.pooling == 'sum':
            n_out = combined_features.sum(dim=2)
    
        
        node_outputs1 = n_out.unsqueeze(1).expand(B, N, N,D)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj,node_outputs1,node_outputs2)
        
        return n_out,edge_outputs

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