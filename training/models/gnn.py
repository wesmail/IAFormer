import torch
import torch.nn as nn


class EEDGCNEncoder(nn.Module):
    def __init__(
        self,
        in_channels_n,
        out_channels_n,
        in_channels_E,
        out_channels_E,
        k,
        n_layers,
        pooling="avg",
    ):
        super(EEDGCNEncoder, self).__init__()
        self.in_channels_n = in_channels_n
        self.out_channels_n = out_channels_n
        self.out_channels_E = out_channels_E
        self.in_channels_E = in_channels_E
        self.k = k
        self.n_layers = n_layers
        self.pooling = pooling
        self.layers = nn.ModuleList(
            [
                EdgeConvWithEdgeFeatures(
                    self.in_channels_n,
                    self.out_channels_n,
                    self.out_channels_E,
                    self.k,
                    self.pooling,
                )
                for i in range(self.n_layers)
            ]
        )
        self.nH = nn.LayerNorm(self.in_channels_n)
        self.nE = nn.LayerNorm(self.in_channels_E)

    def forward(self, H, E):

        out_H = self.nH(H)
        out_E = self.nE(E)

        for layer in self.layers:
            out_H, out_E = layer(out_H, out_E)

        return out_H, out_E


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

        node = torch.cat([node1, node2], dim=-1)  # [batch, node, node, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, node,npde, dim_e]


class EdgeConvWithEdgeFeatures(nn.Module):
    def __init__(self, in_channels, out_channels_n, out_channels_E, k, pooling="avg"):
        super(EdgeConvWithEdgeFeatures, self).__init__()
        self.k = k
        self.pooling = pooling
        self.in_channels = in_channels
        self.out_channels_n = out_channels_n
        self.out_channels_E = out_channels_E
        self.W = nn.Linear(self.in_channels, self.out_channels_E)
        self.highway = Edgeupdate(
            self.in_channels, self.out_channels_E, dropout_ratio=0.2
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.in_channels, self.out_channels_n, bias=False),
            nn.BatchNorm1d(self.out_channels_n),
            nn.GELU(),
        )

    def forward(self, x, weight_adj):
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
            idx.unsqueeze(-1).expand(-1, -1, -1, D),
        )  # [B, N, k, D]

        # Central point repeated for k neighbors: [B, N, k, D]
        central = x.unsqueeze(2).expand(-1, -1, self.k, -1)

        # Step 3: Compute edge features
        relative_features = neighbors - central  # [B, N, k, D]
        combined_features = torch.cat(
            [central, relative_features], dim=-1
        )  # [B, N, k, 2*D + E]

        # Step 4: Apply MLP and aggregation
        combined_features = self.mlp(
            combined_features.view(-1, 2 * D)
        )  # [B * N * k, out_channels]
        combined_features = combined_features.view(
            B, N, self.k, -1
        )  # Reshape to [B, N, k, out_channels]

        if self.pooling == "avg":
            n_out = combined_features.mean(dim=2)
        elif self.pooling == "max":
            n_out = combined_features.max(dim=2)[0]
        elif self.pooling == "sum":
            n_out = combined_features.sum(dim=2)

        node_outputs1 = n_out.unsqueeze(1).expand(B, N, N, D)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        return n_out, edge_outputs


# ***********************************************************************************************************
# LorentzNet
# ***********************************************************************************************************
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter


def normsq4(p, metric=torch.tensor([1, -1, -1, -1])):
    return torch.sum(p**2 * metric, dim=-1, keepdim=True)  # [N, 1]


def dotsq4(p, q, metric=torch.tensor([1, -1, -1, -1])):
    return torch.sum(p * q * metric, dim=-1, keepdim=True)  # [N, 1]


def psi(p):
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


def minkowski_feats(edge_index, x):
    row, col = edge_index
    x_i, x_j = x[row], x[col]
    x_diff = x_i - x_j
    norms = psi(normsq4(x_diff))  # [E, 1]
    dots = psi(dotsq4(x_i, x_j))  # [E, 1]
    return norms, dots


class LGEB(MessagePassing):
    def __init__(self, n_hidden=32, c=0.001, aggr="add"):
        super().__init__(aggr=aggr)
        self.c = c

        # ϕ_e: Edge message network (input: 2 features [norms, dots], output: 1)
        self.phi_e = nn.Sequential(
            nn.Linear(2, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        # ϕ_x: Scalar function for vector update (output: 1)
        self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)

        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())

    def forward(self, x, edge_index):
        # Compute edge features
        norms, dots = minkowski_feats(edge_index, x)

        # Construct edge messages (m_ij)
        mij_input = torch.cat([norms, dots], dim=1)  # [E, 2]
        mij = self.phi_e(mij_input)  # [E, 1]
        w = self.phi_m(mij)
        mij = mij * w

        # Update vectors x via message passing
        x_out = self.propagate(edge_index, x=x, mij=mij)
        return x_out

    def message(self, x_i, x_j, mij):
        # Compute x_diff = x_i - x_j (equivariant term)
        x_diff = x_i - x_j  # [E, 4]
        # Scale x_diff by ϕ_x(mij)
        scale = self.phi_x(mij)  # [E, 1]
        return scale * x_diff  # [N, 4]

    def update(self, aggr_out, x):
        # x_new = x + c * aggregated messages
        return x + self.c * aggr_out  # L 60 in models.py


class LorentzNet(nn.Module):
    def __init__(self, n_hidden=32, num_layers=1, dropout=0.2, c=0.001):
        super().__init__()
        # Stack of LGEB layers
        self.layers = nn.ModuleList(
            [LGEB(n_hidden=n_hidden, c=c) for _ in range(num_layers)]
        )

        # Decoder: Processes mean of final 4-momenta
        self.decoder = nn.Sequential(
            nn.Linear(4, n_hidden),  # Input: averaged 4-vector
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, edge_index):
        # Process through LGEB layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Decoder: Permutation-invariant mean pooling
        x = x.mean(dim=0)  # [4]
        return self.decoder(x)


class LorentzNet(nn.Module):
    def __init__(self, n_hidden=32, num_layers=3, dropout=0.2, c=0.001):
        super().__init__()
        self.layers = nn.ModuleList(
            [LGEB(n_hidden=n_hidden, c=c) for _ in range(num_layers)]
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        # Process through LGEB layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Mean pooling PER GRAPH
        x_avg = scatter(x, batch, dim=0, reduce="mean")  # Uses batch indices
        return self.decoder(x_avg)


# ***********************************************************************************************************
