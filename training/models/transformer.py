import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x, u=None, umask=None):
        x_res = x
        x = self.norm1(x)
        attn_output = self.pmha(x, u, umask)  # x, and u embeddings
        x = self.norm2(attn_output)
        h = x + x_res
        z = self.mlp(h)
        return z + h


class DynamicEdgeConv(MessagePassing):
    def __init__(self, in_channels=2, embed_dim=32, k=6, c_weight=0.001):
        super().__init__(aggr="mean")  #  "Max" aggregation.
        self.k = k
        # self.c_weight = c_weight
        self.phi_e = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

        self.phi_m = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

        layer = nn.Linear(embed_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.phi_x = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), layer)

        self.register_buffer("metric", torch.tensor([1, -1, -1, -1]))

    def normsq4(self, p):
        return torch.sum(p**2 * self.metric, dim=-1, keepdim=True)  # [N, 1]

    def dotsq4(self, p, q):
        return torch.sum(p * q * self.metric, dim=-1, keepdim=True)  # [N, 1]

    def psi(self, p):
        return torch.sign(p) * torch.log(torch.abs(p) + 1)

    def minkowski_feats(self, x, edge_index):
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        x_diff = x_i - x_j
        norms = self.psi(self.normsq4(x_diff))  # [E, 1]
        dots = self.psi(self.dotsq4(x_i, x_j))  # [E, 1]
        return norms, dots

    def forward(self, x, edge_index, batch=None):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return self.propagate(edge_index, x=x)

    def update(self, aggr_out, x):
        # x_new = x + c * aggregated messages
        return x + aggr_out  # L 60 in models.py

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        x_diff = x_i - x_j
        norms = self.psi(self.normsq4(x_diff))  # [E, 1]
        dots = self.psi(self.dotsq4(x_i, x_j))  # [E, 1]

        out = torch.cat([norms, dots], dim=1)  # tmp has shape [E, 2]
        out = self.phi_e(out)
        w = self.phi_m(out)
        out = out * w
        scale = self.phi_x(out)  # [E, 1]
        aggr = scale * x_diff
        return aggr  # [N, 4]


class ParticleNet(nn.Module):
    def __init__(
        self, in_channels=4, num_layers=3, embed_dims=[64, 128, 256], k=[16, 16, 16]
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
                    embed_dim=embed_dims[i],
                    k=k[i],
                )
                for i in range(num_layers)
            ]
        )

        # self.lin = nn.Sequential(
        #    nn.Linear(in_channels, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        # )

    def forward(self, x, edge_index, batch, mask=None):
        # Pass input through each DynamicEdgeConv layer
        for conv in self.edge_conv:
            x = conv(x, edge_index, batch)

        # Aggregate
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # return self.lin(x)

        if mask is not None:
            pad_mask = (mask == 1).sum(dim=-1)
            # Split x into chunks according to the mask
            chunks = torch.split(x, pad_mask.tolist(), dim=0)
            # Pad each chunk to max_b and stack them
            x = torch.nn.utils.rnn.pad_sequence(
                chunks, batch_first=True, padding_value=0
            )

        return x


class ParticleTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_blocks=6,
        k=[16, 32, 32],
        num_classes=1,
    ):
        super(ParticleTransformer, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.u_embed_dim = num_heads * 2
        # self.particle_net = ParticleNet(num_layers=3, embed_dims=[16, 32, 64], k=[4, 8, 16])
        self.particle_embed = ParticleEmbedding(input_dim=4)
        self.interaction_embed = InteractionInputEncoding(
            input_dim=4, output_dim=self.u_embed_dim
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

    def forward(self, x, u=None, pmask=None):
        # x = self.particle_net(x, edge_index, batch, pmask)
        if pmask is not None:
            pad_mask = (pmask == 1).sum(dim=-1)
            x = x[:, : pad_mask.max()]
            u = self.interaction_embed(u[:, : pad_mask.max(), : pad_mask.max(), :])
        x = self.particle_embed(x)
        # num_particles = x.size(1)
        umask = (
            (u == 0)[:, : pad_mask.max(), : pad_mask.max(), 0]
            .bool()
            .unsqueeze(-1)
            .repeat(1, 1, 1, self.u_embed_dim)
        )
        # u = self.interaction_embed(u[:, :num_particles, :num_particles, :])

        for block in self.blocks:
            x = block(x, u, umask)

        # Aggregate features (e.g., mean pooling)
        # TODO: let user choose pooling function
        x = x.mean(dim=1)  # Pool across particles

        logits = self.mlp_head(x)
        return logits


class Embedding(nn.Module):
    """
    Embedding of the feature tokens in both, Transformer and EEDGNN.

    args:
        input_dim: int, dim of the feature tokens
        layers_dim: list, list of the number of neurons of each FC layer

    return:
          MLP network.
    """

    def __init__(self, input_dim, layers_dim):
        super(Embedding, self).__init__()

        self.layers_dim = layers_dim
        self.layers = nn.ModuleList()

        for i in layers_dim[1:]:
            self.layers.append(nn.LayerNorm(input_dim))
            self.layers.append(nn.Linear(input_dim, i))
            self.layers.append(nn.GELU())
            input_dim = i

    def forward(self, input_):
        x = input_
        for layer in self.layers:
            x = layer(x)
        return x


class mlp(nn.Module):
    """
    Final mlp of the network, after pooling the Transformer encoder output.

    args:
        input_dim: int, dim of the feature tokens of Transformer encoder output
        layers_dim: list, list of the number of neurons of each FC layer

    return:
          MLP network with one neuron output and Sigmoid activation
    """

    def __init__(self, input_dim, layers_dim):
        super(mlp, self).__init__()
        self.layers_dim = layers_dim
        self.layers = nn.ModuleList()

        for i in layers_dim[1:]:
            self.layers.append(nn.LayerNorm(input_dim))
            self.layers.append(nn.Linear(input_dim, i))
            self.layers.append(nn.GELU())
            input_dim = i
        self.layers.append(nn.Linear(i, 1))
        self.layers.append(nn.Sigmoid())

    def forward(self, input_):
        x = input_
        for layer in self.layers:
            x = layer(x)
        return x


class MultiHead_Self_Attention(nn.Module):
    def __init__(self, embed_dim0, embed_dim, num_heads, masked=True):
        super(MultiHead_Self_Attention, self).__init__()
        """
        MultiHead self attention with interaction matrix U. 
        The diemsnion of U is (batch,num of heads, particle tokens, particle tokens)
        
        Args:
            embed_dim0: int, dim of the feature tokens
            embed_dim: int,hidden dimension, "scaled attention"
            num_heads: int,number of attention heads
            masked: polean, using of the attention mask to remove the padded points
            
        return:
              1- output of attention heads, with dim (batch,particle tokens, feature tokens)
              2- attention weights, with dim (batch, particle tokens, particle tokens)
        """
        self.masked = masked
        self.embed_dim0 = embed_dim0
        if embed_dim % num_heads == 0:

            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
        else:
            self.embed_dim = embed_dim + (num_heads - embed_dim % num_heads)
            self.num_heads = num_heads
            self.head_dim = self.embed_dim // num_heads

        # Initialize the linear layers
        self.q_linear = nn.Linear(self.embed_dim0, self.embed_dim, bias=False)
        self.k_linear = nn.Linear(self.embed_dim0, self.embed_dim, bias=False)
        self.v_linear = nn.Linear(self.embed_dim0, self.embed_dim, bias=False)
        self.out_linear = nn.Linear(self.embed_dim, self.embed_dim0, bias=False)

    def att_mask(self, input_):
        """
        Function to create attention mask with 1 for unpadded points and 0 for padded points

        arg1: input data set with dim (batch_size, n_particles,n_features)
        output: mask tensor of dim (batch_size, number of heads, n_particles,n_particles)
        """

        mask = (input_.sum(dim=-1) != 0).float()
        mask = mask[:, :, None, :]
        mask = mask.repeat(1, 1, mask.size(-1), 1)

        return mask

    def scaled_dot_product_attention(self, Q, K, V, U):
        """
        Computes scaled dot-product attention.
        dim of U: batch_size x num_heads x particle_tokens x particle_tokens
        """
        d_k = Q.size(-1)

        scores = (torch.matmul(Q, K.transpose(-2, -1)) + U) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )

        if self.masked:
            mask = self.att_mask(Q)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, U):
        batch_size = query.size(0)

        Q = self.q_linear(query)
        K = self.k_linear(query)
        V = self.v_linear(query)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, U)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )

        # Apply final linear layer # W matrix
        output = self.out_linear(attn_output)

        return output, attn_weights


###########################################
###########################################
###########################################
class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        h_dim=500,
        expansion_factor=4,
        n_heads=10,
        masked=True,
    ):
        super(TransformerLayer, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        """
        To be done: 
               Here, the hidden dimension is fixed, may be we need to adopt it in the future.
        """

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.masked = masked
        self.expansion_factor = expansion_factor
        self.h_dim = h_dim
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.attention = MultiHead_Self_Attention(
            self.input_dim, self.h_dim, self.n_heads, self.masked
        )
        self.norm1 = nn.LayerNorm(self.input_dim)
        self.norm2 = nn.LayerNorm(self.input_dim)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.expansion_factor * self.input_dim),
            nn.GELU(),
            nn.LayerNorm(self.expansion_factor * self.input_dim),
            nn.Linear(self.expansion_factor * self.input_dim, self.input_dim),
        )

    def forward(self, query, U):

        attention_out, _ = self.attention(self.norm1(query), U)
        attention_residual_out = self.norm2(attention_out) + query
        norm1_out = self.dropout1(attention_residual_out)
        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(feed_fwd_residual_out)

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention

    Returns:
        out: output of the encoder
    """

    def __init__(
        self,
        input_dim,
        embed_dim=[512, 256, 128],
        h_dim=200,
        num_layers=2,
        expansion_factor=4,
        n_heads=10,
        masked=True,
    ):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.h_dim = h_dim
        self.masked = masked
        self.expansion_factor = expansion_factor
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.embed = Embedding(self.input_dim, self.embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim[-1],
                    self.h_dim,
                    self.expansion_factor,
                    self.n_heads,
                    self.masked,
                )
                for i in range(self.num_layers)
            ]
        )

    def forward(self, x, u):
        x_new = self.embed(x)
        out = F.layer_norm(x_new, x_new.shape)

        for layer in self.layers:
            out = layer(out, u)

        return out
