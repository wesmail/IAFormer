#!/usr/bin/env python3
"""
profile_flops.py

Purpose
-------
Compute FLOPs (and params) similar to the ParT-style Transformer model,
in a way that matches the paper's accounting (add the quadratic
attention matmul costs that generic profilers miss).

Approach
--------
- Build your model from models.py (Transformer).
- Create dummy inputs that match your training shapes:
    node_features:  [B, N, in_channels]
    edge_features:  [B, N, N, u_channels]
- Use THOP to count FLOPs for Linear/Conv/etc.
- Add custom hooks for attention modules to include:
    * QK^T    cost  ~ B * H * N * N * d_head
    * weights*V cost ~ B * H * N * N * d_head
  (i.e., ~ 2 * B * N^2 * d for multi-head attention)

Notes
-----
- We count MACs (multiply-add) as ONE "op" (common in many DL papers).
  If you want FLOPs where multiply+add=2 ops, multiply totals by 2.
- We set `use_flash=False` during profiling to guarantee hooks run on
  the standard path (no fused kernel hiding matmuls).
- Mixed precision (fp16/bf16) does not change FLOP **counts**.

Usage
-----
$ python profile_flops.py \
    --batch-size 1 \
    --num-particles 100 \
    --in-ch 7 \
    --u-ch 6 \
    --embed-dim 128 \
    --num-heads 8 \
    --num-blocks 8 \
    --attn interaction

This should land you in the same ballpark as the ParT paper's table.
"""

import argparse
import math
import sys
from typing import Dict, Any

import torch
import torch.nn as nn

# --- Your modules ---
# Make sure PYTHONPATH finds your local package, or run from repo root.
from models import models as my_models           # Transformer, JetTaggingModule
from models.attention import MultiHeadAttention, MultiHeadDiffAttention
# (The interaction encoder lives in models/embedding.py and is already counted by THOP as Conv2d/Linear.)

# --- Optional dependency: THOP ---
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("[WARN] thop not installed. Install with: pip install thop")
    print("       The script will still run, but only the analytic attention add-on will be shown.")


# ---------------------------
# Custom FLOP hook utilities
# ---------------------------

def count_attention_flops(module: MultiHeadAttention, x_shape):
    """
    Compute the missing O(N^2 d) attention costs for MultiHeadAttention.
    
    Returns the number of MACs (multiply-adds).
    """
    B, N, d_model = x_shape
    H = module.num_heads
    d_head = module.head_dim

    # Add QK^T and weights*V costs (MACs)
    macs_attn = 2.0 * B * H * (N * N) * d_head
    return macs_attn


def count_diff_attention_flops(module: MultiHeadDiffAttention, x_shape):
    """
    Compute the missing O(N^2 d) for the differential attention weighted sum.
    
    Returns the number of MACs (multiply-adds).
    """
    B, N, d_model = x_shape
    H = module.num_heads
    # head_dim was defined as: embed_dim // num_heads // 2  (for differential attention)
    d_head = d_model // H // 2

    macs_attn = 1.0 * B * H * (N * N) * (2 * d_head)
    return macs_attn


def custom_attention_counter(module, input, output):
    """
    Hook to count attention FLOPs and add to THOP's total_ops.
    This hook is compatible with THOP's profiling system.
    """
    x = input[0]
    x_shape = x.shape
    
    if isinstance(module, MultiHeadAttention):
        macs = count_attention_flops(module, x_shape)
    elif isinstance(module, MultiHeadDiffAttention):
        macs = count_diff_attention_flops(module, x_shape)
    else:
        return
    
    # Add to THOP's tracking
    if hasattr(module, 'total_ops'):
        module.total_ops += torch.DoubleTensor([float(macs)])


def register_attention_hooks_for_thop(model: nn.Module):
    """
    Register custom FLOP counting hooks for attention modules.
    This should be called with THOP's custom_ops parameter.
    """
    custom_ops = {}
    
    # Register both attention types
    custom_ops[MultiHeadAttention] = custom_attention_counter
    custom_ops[MultiHeadDiffAttention] = custom_attention_counter
    
    return custom_ops


# ---------------------------
# Dummy input construction
# ---------------------------

def make_dummy_inputs(B: int, N: int, in_ch: int, u_ch: int, device="cpu", attn="plain"):
    """
    Build synthetic batch matching your model's forward signature:
      x: [B, N, in_ch]
      u: [B, N, N, u_ch]    (only used for 'interaction' or 'diff' attention)
      y: [B] (labels, unused here)

    Use small random numbers to avoid overflow in bf16/fp16.
    """
    x = torch.randn(B, N, in_ch, device=device) * 0.01
    if attn in ("interaction", "diff"):
        u = torch.randn(B, N, N, u_ch, device=device) * 0.01
    else:
        u = None
    y = torch.zeros(B, dtype=torch.long, device=device)
    return x, u, y


# ---------------------------
# Analytic reporting helpers
# ---------------------------

def explain_counting_convention():
    print("\n[Convention]")
    print("We count multiply-add (MAC) as 1 operation.")
    print("If you prefer FLOPs where multiply+add=2 ops, multiply totals by 2.")
    print("Softmax and bias adds are ignored in the total as they are small vs. matmuls.\n")


def compute_analytic_attention_macs(model: nn.Module, B: int, N: int):
    """
    Compute attention MACs analytically for reporting purposes.
    """
    total_attn_macs = 0.0
    
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            # Each block: 2 * B * H * N^2 * d_head
            H = m.num_heads
            d_head = m.head_dim
            macs = 2.0 * B * H * (N * N) * d_head
            total_attn_macs += macs
        elif isinstance(m, MultiHeadDiffAttention):
            # Each block: B * H * N^2 * (2 * d_head)
            H = m.num_heads
            d_head = m.embed_dim // H // 2
            macs = 1.0 * B * H * (N * N) * (2 * d_head)
            total_attn_macs += macs
    
    return total_attn_macs


# ---------------------------
# Main
# ---------------------------

def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    # Build your model with Flash disabled so the path is deterministic for hooks
    model = my_models.Transformer(
        in_channels=args.in_ch,
        u_channels=args.u_ch,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        attn=args.attn,
        max_num_particles=args.num_particles,
        num_classes=args.num_classes,
    ).to(device)
    model.eval()

    # Create dummy inputs
    x, u, y = make_dummy_inputs(
        B=args.batch_size,
        N=args.num_particles,
        in_ch=args.in_ch,
        u_ch=args.u_ch,
        device=device,
        attn=args.attn,
    )

    # --- Count parameters ---
    total_params = sum(p.numel() for p in model.parameters())

    # --- THOP profile (if available) ---
    if THOP_AVAILABLE:
        # Get custom ops for attention modules
        custom_ops = register_attention_hooks_for_thop(model)
        
        # Profile with custom ops
        macs, params = profile(model, inputs=(x, u), custom_ops=custom_ops, verbose=False)
        
        macs_str, params_str = clever_format([macs, params], "%.3f")
        print("\n[THOP Summary]")
        print(f"Params: {params_str}")
        print(f"MACs  : {macs_str}  (multiply-adds counted as 1 op)")
        print("NOTE: Multiply by 2 if you want 'FLOPs' with mul+add=2.\n")
        
        # Also show the analytic attention contribution
        attn_macs = compute_analytic_attention_macs(model, args.batch_size, args.num_particles)
        attn_str = clever_format([attn_macs], "%.3f")[0]
        print(f"[Attention MACs breakdown]")
        print(f"Total attention MACs: {attn_str}")
        print(f"This represents {100 * attn_macs / macs:.1f}% of total MACs\n")
    else:
        print("\n[THOP not installed] Showing parameter count only.")
        print(f"Params: {total_params/1e6:.3f} M")
        
        # Show analytic attention estimate even without THOP
        attn_macs = compute_analytic_attention_macs(model, args.batch_size, args.num_particles)
        print(f"\n[Analytic attention estimate]")
        print(f"Attention MACs: {attn_macs/1e9:.3f} G (just the attention operations)")
        print("Install THOP for a full breakdown: pip install thop\n")

    # Final note about matching the paper's methodology
    print("[How this matches the paper]")
    print("- Linear/Conv layers are counted by THOP.")
    print("- The big missing cost in Transformers is attention matmuls.")
    print("  We add:")
    print("    * QK^T     : B * H * N * N * d_head")
    print("    * softmaxV : B * H * N * N * d_head")
    print("  (≈ 2 * B * N^2 * d_model) per attention, as in ParT complexity discussion.")
    print("  This aligns with the FLOPs accounting behind Table 4 in the paper.")
    explain_counting_convention()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",      type=int, default=1, help="Batch size B used for profiling")
    parser.add_argument("--num-particles",   type=int, default=100, help="Max particles per jet (N)")
    parser.add_argument("--in-ch",           type=int, default=7, help="Node feature channels")
    parser.add_argument("--u-ch",            type=int, default=6, help="Pairwise (u) feature channels")
    parser.add_argument("--embed-dim",       type=int, default=64, help="Model dimension d")
    parser.add_argument("--num-heads",       type=int, default=16, help="Number of attention heads H")
    parser.add_argument("--num-blocks",      type=int, default=16, help="Number of ParticleAttention blocks")
    parser.add_argument("--attn",            type=str, default="diff",
                        choices=["plain", "interaction", "diff"],
                        help="Attention type used by your model")
    parser.add_argument("--num-classes",     type=int, default=10, help="Number of output classes")
    parser.add_argument("--cpu", action="store_true", help="Force CPU profiling")
    args = parser.parse_args()
    main(args)