"""
Fused Triton kernels for LlamaDecoderLayer: RMSNorm, QKV projection, MLP.

Reduces per-layer GPU kernel count from ~34 to ~17:
  - RMSNorm: 5 eager kernels → 1 Triton kernel (×2 per layer)
  - QKV projection: 3 matmuls → 1 fused matmul
  - MLP gate+up: 2 matmuls → 1 fused matmul
  - SiLU+mul: 2 eager kernels → 1 Triton kernel (reads from fused output, no .contiguous())

With 30 layers: ~1020 kernels/token → ~510 kernels/token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ─── Triton kernels ────────────────────────────────────────────────

if HAS_TRITON:
    @triton.jit
    def _rms_norm_kernel(
        x_ptr, weight_ptr, out_ptr,
        N_cols: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Fused RMSNorm: cast→variance→rsqrt→scale in one pass."""
        row = tl.program_id(0)
        offs = tl.arange(0, N_cols)

        x = tl.load(x_ptr + row * N_cols + offs).to(tl.float32)
        w = tl.load(weight_ptr + offs).to(tl.float32)

        var = tl.sum(x * x, axis=0) / N_cols
        rrms = 1.0 / tl.sqrt(var + eps)
        result = x * rrms * w

        tl.store(out_ptr + row * N_cols + offs, result.to(tl.bfloat16))

    @triton.jit
    def _silu_mul_fused_kernel(
        gate_up_ptr, out_ptr,
        stride_row,
        intermediate_size: tl.constexpr,
    ):
        """Fused SiLU(gate) * up reading directly from concatenated gate_up tensor.
        No .contiguous() copy needed — reads gate at [row, 0:D] and up at [row, D:2D].
        """
        row = tl.program_id(0)
        offs = tl.arange(0, intermediate_size)

        base = row * stride_row
        gate = tl.load(gate_up_ptr + base + offs).to(tl.float32)
        up = tl.load(gate_up_ptr + base + intermediate_size + offs).to(tl.float32)

        result = gate * tl.sigmoid(gate) * up

        tl.store(out_ptr + row * intermediate_size + offs, result.to(tl.bfloat16))


# ─── Fused RMSNorm ─────────────────────────────────────────────────

class FusedRMSNorm(nn.Module):
    """Drop-in replacement for LlamaRMSNorm using a single Triton kernel."""

    def __init__(self, original_norm: nn.Module):
        super().__init__()
        self.weight = original_norm.weight
        self.variance_epsilon = original_norm.variance_epsilon
        self.hidden_size = self.weight.shape[0]

    def forward(self, hidden_states):
        if HAS_TRITON and hidden_states.is_cuda and hidden_states.dtype == torch.bfloat16:
            x = hidden_states.view(-1, self.hidden_size)
            out = torch.empty_like(x)
            _rms_norm_kernel[(x.shape[0],)](
                x, self.weight, out,
                N_cols=self.hidden_size,
                eps=self.variance_epsilon,
            )
            return out.view_as(hidden_states)
        else:
            # Fallback
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


# ─── Fused QKV Projection ──────────────────────────────────────────

class FusedQKVProj(nn.Module):
    """Fuses q_proj + k_proj + v_proj into a single matmul."""

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear):
        super().__init__()
        q_dim = q_proj.weight.shape[0]
        k_dim = k_proj.weight.shape[0]
        v_dim = v_proj.weight.shape[0]
        hidden = q_proj.weight.shape[1]

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        # Concatenate weights: [q_dim + k_dim + v_dim, hidden]
        fused_weight = torch.cat([
            q_proj.weight.data,
            k_proj.weight.data,
            v_proj.weight.data,
        ], dim=0)
        self.qkv_proj = nn.Linear(hidden, q_dim + k_dim + v_dim, bias=False)
        self.qkv_proj.weight = nn.Parameter(fused_weight)

    def forward(self, hidden_states):
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_dim]
        k = qkv[..., self.q_dim:self.q_dim + self.k_dim]
        v = qkv[..., self.q_dim + self.k_dim:]
        return q, k, v


# ─── Fused MLP ─────────────────────────────────────────────────────

class FusedLlamaMLP(nn.Module):
    """Fused gate_up projection + SiLU*mul Triton kernel. No .contiguous() copies."""

    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        self.hidden_size = original_mlp.hidden_size
        self.intermediate_size = original_mlp.intermediate_size

        gate_weight = original_mlp.gate_proj.weight.data
        up_weight = original_mlp.up_proj.weight.data
        fused_weight = torch.cat([gate_weight, up_weight], dim=0)
        self.gate_up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.gate_up_proj.weight = nn.Parameter(fused_weight)
        self.down_proj = original_mlp.down_proj

    def forward(self, x):
        gate_up = self.gate_up_proj(x)  # [batch, seq, 2*intermediate]

        if HAS_TRITON and gate_up.is_cuda and gate_up.dtype == torch.bfloat16:
            shape = gate_up.shape[:-1]  # [batch, seq]
            n_rows = gate_up[..., 0].numel()
            intermediate = torch.empty(
                *shape, self.intermediate_size,
                dtype=gate_up.dtype, device=gate_up.device,
            )
            gate_up_flat = gate_up.view(-1, 2 * self.intermediate_size)
            _silu_mul_fused_kernel[(n_rows,)](
                gate_up_flat, intermediate.view(-1, self.intermediate_size),
                stride_row=2 * self.intermediate_size,
                intermediate_size=self.intermediate_size,
            )
        else:
            gate = gate_up[..., :self.intermediate_size]
            up = gate_up[..., self.intermediate_size:]
            intermediate = F.silu(gate) * up

        return self.down_proj(intermediate)


# ─── Replacement utility ───────────────────────────────────────────

def fuse_decoder_layers(model: nn.Module) -> dict:
    """Replace LlamaMLP, RMSNorm, and QKV projections with fused versions.

    Returns dict with counts of replaced components.
    """
    if not HAS_TRITON:
        return {"mlp": 0, "rmsnorm": 0, "qkv": 0}

    counts = {"mlp": 0, "rmsnorm": 0, "qkv": 0}

    for layer in model.layers:
        # Fuse MLP
        if hasattr(layer, 'mlp'):
            layer.mlp = FusedLlamaMLP(layer.mlp)
            counts["mlp"] += 1

        # Fuse RMSNorm (input + post_attention)
        if hasattr(layer, 'input_layernorm'):
            layer.input_layernorm = FusedRMSNorm(layer.input_layernorm)
            counts["rmsnorm"] += 1
        if hasattr(layer, 'post_attention_layernorm'):
            layer.post_attention_layernorm = FusedRMSNorm(layer.post_attention_layernorm)
            counts["rmsnorm"] += 1

        # Fuse QKV projections
        attn = layer.self_attn
        if hasattr(attn, 'q_proj') and hasattr(attn, 'k_proj') and hasattr(attn, 'v_proj'):
            attn._fused_qkv = FusedQKVProj(attn.q_proj, attn.k_proj, attn.v_proj)
            counts["qkv"] += 1

    # Also fuse the final RMSNorm
    if hasattr(model, 'norm'):
        model.norm = FusedRMSNorm(model.norm)
        counts["rmsnorm"] += 1

    return counts
