#!/usr/bin/env python3
"""
FLOPs/MACs profiler for SparseTransformer using THOP.
We estimate sparse attention cost as 2 * E * H * d_head MACs per layer
(QK^T on edges and weights*V), in addition to Linear layers counted by THOP.
"""

import torch
from thop import profile, clever_format

from models.models_sparse import SparseTransformer, SparseMultiHeadAttention, SparseMultiHeadDiffAttention
from models.models import Transformer as DenseTransformer
from torch_geometric.data import Batch, Data


def build_synthetic_batch(batch_size=2, num_nodes_per_graph=32, in_channels=11, edge_dim=6, device="cpu"):
    """Create a simple batched graph with upper-triangular edges per graph."""
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_vec = []

    node_offset = 0
    for g in range(batch_size):
        N = num_nodes_per_graph
        x_g = torch.randn(N, in_channels, device=device)
        # upper triangular edges (i < j)
        i, j = torch.triu_indices(N, N, offset=1, device=device)
        edge_index_g = torch.stack([i + node_offset, j + node_offset], dim=0)
        E = edge_index_g.size(1)
        edge_attr_g = torch.randn(E, edge_dim, device=device)

        x_list.append(x_g)
        edge_index_list.append(edge_index_g)
        edge_attr_list.append(edge_attr_g)
        batch_vec.append(torch.full((N,), g, dtype=torch.long, device=device))

        node_offset += N

    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    batch = torch.cat(batch_vec, dim=0)
    return x, edge_index, edge_attr, batch


def sparse_attention_custom_op(module, inputs, output):
    """THOP custom op to count sparse attention edge-wise MACs.
    Counts 2 * E * H * d_head (QK dot + weights*V).
    """
    # inputs: (x, edge_index, edge_attr, batch?)
    if not inputs or len(inputs) < 2:
        return
    edge_index = inputs[1]
    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2:
        return
    E = edge_index.size(1)
    H = getattr(module, "num_heads", None)
    d_head = getattr(module, "head_dim", None)
    if H is None or d_head is None:
        return
    macs = 2.0 * E * H * d_head
    if hasattr(module, "total_ops"):
        module.total_ops += torch.DoubleTensor([float(macs)])


def sparse_diff_attention_custom_op(module, inputs, output):
    """THOP custom op to count sparse diff attention edge-wise MACs.
    Counts 4 * E * H * d_head (two attention maps: QK dot + weights*V for each).
    """
    # inputs: (x, edge_index, edge_attr, batch?)
    if not inputs or len(inputs) < 2:
        return
    edge_index = inputs[1]
    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2:
        return
    E = edge_index.size(1)
    H = getattr(module, "num_heads", None)
    d_head = getattr(module, "head_dim", None)
    if H is None or d_head is None:
        return
    # Diff attention computes two attention maps, so 2x the cost
    macs = 4.0 * E * H * d_head
    if hasattr(module, "total_ops"):
        module.total_ops += torch.DoubleTensor([float(macs)])


def build_custom_ops_dict(model):
    return {
        SparseMultiHeadAttention: sparse_attention_custom_op,
        SparseMultiHeadDiffAttention: sparse_diff_attention_custom_op,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 80)
    print("FLOPs/MACs Profiler: Sparse vs Dense Transformer Comparison")
    print("=" * 80)
    print()

    # Common parameters
    in_ch = 11
    edge_ch = 6
    embed_dim = 64
    num_heads = 16
    num_blocks = 16
    num_classes = 10
    
    # Sparse model parameters
    num_nodes_per_graph = 32
    
    # Dense model parameters
    B = 1
    P = 100  # max_num_particles

    # Build sparse batch
    x_sparse, edge_index, edge_attr, batch_vec = build_synthetic_batch(
        batch_size=1, num_nodes_per_graph=num_nodes_per_graph, 
        in_channels=in_ch, edge_dim=edge_ch, device=device
    )
    
    # Create PyG Batch object for SparseTransformer
    data = Data(x=x_sparse, edge_index=edge_index, edge_attr=edge_attr, batch=batch_vec)
    batch_sparse = Batch.from_data_list([data])
    
    # Build dense batch
    x_dense = torch.randn(B, P, in_ch, device=device)
    u_dense = torch.randn(B, P, P, edge_ch, device=device)

    custom_ops = build_custom_ops_dict(None)  # Pass None since we build the dict separately

    results = {}

    # ============================================================
    # 1. SparseTransformer with Plain Attention
    # ============================================================
    print("[1/4] SparseTransformer (plain attention)")
    print("-" * 80)
    sparse_plain = SparseTransformer(
        in_channels=in_ch,
        edge_channels=edge_ch,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        attn="plain",
        num_classes=num_classes,
    ).to(device)
    sparse_plain.eval()
    
    macs_sp, params_sp = profile(
        sparse_plain, 
        inputs=(batch_sparse,), 
        custom_ops=custom_ops, 
        verbose=False
    )
    results['sparse_plain'] = {'macs': macs_sp, 'params': params_sp}
    
    macs_str, params_str = clever_format([macs_sp, params_sp], "%.3f")
    print(f"Params: {params_str}")
    print(f"MACs  : {macs_str}  (1 MAC = 1 multiply+add)")
    flops = 2 * macs_sp
    flops_str = clever_format([flops], "%.3f")
    print(f"FLOPs : {flops_str}  (1 FLOP = 1 multiply+add)")
    print()

    # ============================================================
    # 2. SparseTransformer with Diff Attention
    # ============================================================
    print("[2/4] SparseTransformer (diff attention)")
    print("-" * 80)
    sparse_diff = SparseTransformer(
        in_channels=in_ch,
        edge_channels=edge_ch,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        attn="diff",
        num_classes=num_classes,
    ).to(device)
    sparse_diff.eval()
    
    macs_sd, params_sd = profile(
        sparse_diff, 
        inputs=(batch_sparse,), 
        custom_ops=custom_ops, 
        verbose=False
    )
    results['sparse_diff'] = {'macs': macs_sd, 'params': params_sd}
    
    macs_str, params_str = clever_format([macs_sd, params_sd], "%.3f")
    print(f"Params: {params_str}")
    print(f"MACs  : {macs_str}  (1 MAC = 1 multiply+add)")
    flops = 2 * macs_sd
    flops_str = clever_format([flops], "%.3f")
    print(f"FLOPs : {flops_str}  (1 FLOP = 1 multiply+add)")
    print()

    # ============================================================
    # 3. DenseTransformer with Plain Attention
    # ============================================================
    print("[3/4] DenseTransformer (plain attention)")
    print("-" * 80)
    dense_plain = DenseTransformer(
        in_channels=in_ch,
        u_channels=edge_ch,
        attn="plain",
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        max_num_particles=P,
        num_classes=num_classes,
        use_flash=False,
    ).to(device)
    dense_plain.eval()

    macs_dp, params_dp = profile(
        dense_plain, 
        inputs=(x_dense, u_dense), 
        verbose=False
    )
    results['dense_plain'] = {'macs': macs_dp, 'params': params_dp}
    
    macs_str, params_str = clever_format([macs_dp, params_dp], "%.3f")
    print(f"Params: {params_str}")
    print(f"MACs  : {macs_str}  (1 MAC = 1 multiply+add)")
    flops = 2 * macs_dp
    flops_str = clever_format([flops], "%.3f")
    print(f"FLOPs : {flops_str}  (1 FLOP = 1 multiply+add)")
    print()

    # ============================================================
    # 4. DenseTransformer with Diff Attention
    # ============================================================
    print("[4/4] DenseTransformer (diff attention)")
    print("-" * 80)
    dense_diff = DenseTransformer(
        in_channels=in_ch,
        u_channels=edge_ch,
        attn="diff",
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        max_num_particles=P,
        num_classes=num_classes,
        use_flash=False,
    ).to(device)
    dense_diff.eval()

    macs_dd, params_dd = profile(
        dense_diff, 
        inputs=(x_dense, u_dense), 
        verbose=False
    )
    results['dense_diff'] = {'macs': macs_dd, 'params': params_dd}
    
    macs_str, params_str = clever_format([macs_dd, params_dd], "%.3f")
    print(f"Params: {params_str}")
    print(f"MACs  : {macs_str}  (1 MAC = 1 multiply+add)")
    flops = 2 * macs_dd
    flops_str = clever_format([flops], "%.3f")
    print(f"FLOPs : {flops_str}  (1 FLOP = 1 multiply+add)")
    print()

    # ============================================================
    # Comparison Summary
    # ============================================================
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()
    
    # MACs comparison
    print("MACs Comparison:")
    print(f"  Sparse (plain):  {clever_format([results['sparse_plain']['macs']], '%.3f')[0]}")
    print(f"  Sparse (diff):   {clever_format([results['sparse_diff']['macs']], '%.3f')[0]}")
    print(f"  Dense (plain):   {clever_format([results['dense_plain']['macs']], '%.3f')[0]}")
    print(f"  Dense (diff):    {clever_format([results['dense_diff']['macs']], '%.3f')[0]}")
    print()
    
    # Speedup ratios
    print("Speedup Ratios (vs Dense Plain):")
    speedup_sp = results['dense_plain']['macs'] / results['sparse_plain']['macs']
    speedup_sd = results['dense_plain']['macs'] / results['sparse_diff']['macs']
    speedup_dd = results['dense_plain']['macs'] / results['dense_diff']['macs']
    print(f"  Sparse (plain):  {speedup_sp:.2f}x")
    print(f"  Sparse (diff):   {speedup_sd:.2f}x")
    print(f"  Dense (diff):    {speedup_dd:.2f}x")
    print()
    
    # Parameters comparison
    print("Parameters Comparison:")
    print(f"  Sparse (plain):  {clever_format([results['sparse_plain']['params']], '%.3f')[0]}")
    print(f"  Sparse (diff):   {clever_format([results['sparse_diff']['params']], '%.3f')[0]}")
    print(f"  Dense (plain):   {clever_format([results['dense_plain']['params']], '%.3f')[0]}")
    print(f"  Dense (diff):    {clever_format([results['dense_diff']['params']], '%.3f')[0]}")
    print()
    
    # Diff vs Plain overhead
    print("Diff Attention Overhead (vs Plain):")
    sparse_diff_overhead = (results['sparse_diff']['macs'] / results['sparse_plain']['macs'] - 1) * 100
    dense_diff_overhead = (results['dense_diff']['macs'] / results['dense_plain']['macs'] - 1) * 100
    print(f"  Sparse: +{sparse_diff_overhead:.1f}%")
    print(f"  Dense:  +{dense_diff_overhead:.1f}%")
    print()    


if __name__ == "__main__":
    main()
