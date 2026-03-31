"""
Fused Triton kernels for LlamaDecoderLayer.

Kernel reductions per layer:
  - RMSNorm: 5 eager → 1 Triton (×2 per layer + 1 final = -248)
  - Residual + RMSNorm: 2 ops → 1 Triton (×2 per layer = -60)
  - QKV projection: 3 matmuls → 1 fused matmul (-60)
  - RoPE: ~10 eager → 1 Triton for Q+K combined (×30 = -270)
  - MLP gate+up: 2 matmuls → 1 fused matmul + SiLU*mul in 1 Triton (-90)

With 30 layers: ~1020 kernels/token → ~300 kernels/token.
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
    def _residual_rms_norm_kernel(
        x_ptr, residual_ptr, weight_ptr, out_ptr, residual_out_ptr,
        N_cols: tl.constexpr,
        eps: tl.constexpr,
    ):
        """Fused residual add + RMSNorm in one pass.
        residual_out = x + residual
        out = rmsnorm(residual_out)
        Saves 1 kernel (add) + reads/writes of intermediate tensor.
        """
        row = tl.program_id(0)
        offs = tl.arange(0, N_cols)

        x = tl.load(x_ptr + row * N_cols + offs).to(tl.float32)
        res = tl.load(residual_ptr + row * N_cols + offs).to(tl.float32)
        w = tl.load(weight_ptr + offs).to(tl.float32)

        # Residual add
        hidden = x + res
        tl.store(residual_out_ptr + row * N_cols + offs, hidden.to(tl.bfloat16))

        # RMSNorm
        var = tl.sum(hidden * hidden, axis=0) / N_cols
        rrms = 1.0 / tl.sqrt(var + eps)
        result = hidden * rrms * w

        tl.store(out_ptr + row * N_cols + offs, result.to(tl.bfloat16))

    @triton.jit
    def _silu_mul_fused_kernel(
        gate_up_ptr, out_ptr,
        stride_row,
        intermediate_size: tl.constexpr,
    ):
        """Fused SiLU(gate) * up reading directly from concatenated gate_up tensor."""
        row = tl.program_id(0)
        offs = tl.arange(0, intermediate_size)

        base = row * stride_row
        gate = tl.load(gate_up_ptr + base + offs).to(tl.float32)
        up = tl.load(gate_up_ptr + base + intermediate_size + offs).to(tl.float32)

        result = gate * tl.sigmoid(gate) * up

        tl.store(out_ptr + row * intermediate_size + offs, result.to(tl.bfloat16))

    @triton.jit
    def _rotary_emb_kernel(
        q_ptr, k_ptr, cos_ptr, sin_ptr,
        q_out_ptr, k_out_ptr,
        seq_stride_q, seq_stride_k,
        head_stride_q, head_stride_k,
        HALF_DIM: tl.constexpr,
    ):
        """Fused RoPE for Q and K: rotate_half + mul + add in one kernel.

        Grid: (batch * num_heads * seq_len,)
        Each program processes one (batch, head, seq_pos) pair for both Q and K.

        Q shape: [batch, heads, seq, head_dim] contiguous
        K shape: [batch, heads, seq, head_dim] contiguous
        cos/sin: [batch, 1, seq, head_dim] — broadcast over heads
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, HALF_DIM)

        # Q pointers
        q_base = pid * head_stride_q
        q_first = tl.load(q_ptr + q_base + offs).to(tl.float32)
        q_second = tl.load(q_ptr + q_base + HALF_DIM + offs).to(tl.float32)

        # K pointers
        k_base = pid * head_stride_k
        k_first = tl.load(k_ptr + k_base + offs).to(tl.float32)
        k_second = tl.load(k_ptr + k_base + HALF_DIM + offs).to(tl.float32)

        # cos/sin — broadcast: compute the cos/sin index from pid
        # pid = batch * heads * seq_pos → we need batch * seq_pos (skip heads)
        # cos layout: [batch, 1, seq, head_dim] stored as [batch*seq, head_dim]
        cos_base = pid * seq_stride_q
        c = tl.load(cos_ptr + cos_base + offs).to(tl.float32)
        s = tl.load(sin_ptr + cos_base + offs).to(tl.float32)
        c2 = tl.load(cos_ptr + cos_base + HALF_DIM + offs).to(tl.float32)
        s2 = tl.load(sin_ptr + cos_base + HALF_DIM + offs).to(tl.float32)

        # RoPE: q_embed = q * cos + rotate_half(q) * sin
        # rotate_half([x1, x2]) = [-x2, x1]
        # First half output:  q1 * cos1 + (-q2) * sin1
        # Second half output: q2 * cos2 + q1 * sin2
        q_out_first = q_first * c + (-q_second) * s
        q_out_second = q_second * c2 + q_first * s2
        k_out_first = k_first * c + (-k_second) * s
        k_out_second = k_second * c2 + k_first * s2

        tl.store(q_out_ptr + q_base + offs, q_out_first.to(tl.bfloat16))
        tl.store(q_out_ptr + q_base + HALF_DIM + offs, q_out_second.to(tl.bfloat16))
        tl.store(k_out_ptr + k_base + offs, k_out_first.to(tl.bfloat16))
        tl.store(k_out_ptr + k_base + HALF_DIM + offs, k_out_second.to(tl.bfloat16))


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
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


# ─── Fused Residual + RMSNorm ──────────────────────────────────────

def fused_residual_rmsnorm(x, residual, weight, eps, hidden_size):
    """Fused residual add + RMSNorm. Returns (normed_output, new_residual)."""
    if HAS_TRITON and x.is_cuda and x.dtype == torch.bfloat16:
        x_flat = x.view(-1, hidden_size)
        res_flat = residual.view(-1, hidden_size)
        out = torch.empty_like(x_flat)
        new_residual = torch.empty_like(x_flat)
        _residual_rms_norm_kernel[(x_flat.shape[0],)](
            x_flat, res_flat, weight, out, new_residual,
            N_cols=hidden_size,
            eps=eps,
        )
        return out.view_as(x), new_residual.view_as(x)
    else:
        new_residual = x + residual
        x32 = new_residual.to(torch.float32)
        variance = x32.pow(2).mean(-1, keepdim=True)
        normed = weight * (x32 * torch.rsqrt(variance + eps)).to(x.dtype)
        return normed, new_residual


# ─── Fused RoPE ────────────────────────────────────────────────────

def fused_apply_rotary_pos_emb(q, k, cos, sin):
    """Fused RoPE application for Q and K in one Triton kernel.

    Args:
        q: [batch, heads, seq, head_dim]
        k: [batch, heads, seq, head_dim]
        cos: [batch, seq, head_dim] (3D from LlamaRotaryEmbedding)
        sin: [batch, seq, head_dim] (3D)

    Returns:
        (q_rotated, k_rotated) same shapes
    """
    batch, heads, seq_len, head_dim = q.shape
    half_dim = head_dim // 2

    if not (HAS_TRITON and q.is_cuda and q.dtype == torch.bfloat16):
        # Eager fallback — same as original apply_rotary_pos_emb
        from .modeling_llama import apply_rotary_pos_emb, rotate_half
        return apply_rotary_pos_emb(q, k, cos, sin)

    total = batch * heads * seq_len

    q_c = q.contiguous()
    k_c = k.contiguous()
    q_out = torch.empty_like(q_c)
    k_out = torch.empty_like(k_c)

    # cos/sin: [batch, seq, head_dim] → unsqueeze(1) + expand → [batch, heads, seq, head_dim]
    if cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    cos_exp = cos.expand(batch, heads, seq_len, head_dim).contiguous()
    sin_exp = sin.expand(batch, heads, seq_len, head_dim).contiguous()

    _rotary_emb_kernel[(total,)](
        q_c, k_c, cos_exp, sin_exp,
        q_out, k_out,
        seq_stride_q=head_dim,  # cos stride matches q stride (after expand)
        seq_stride_k=head_dim,
        head_stride_q=head_dim,
        head_stride_k=head_dim,
        HALF_DIM=half_dim,
    )

    return q_out, k_out


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
    """Fused gate_up projection + SiLU*mul Triton kernel."""

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
        gate_up = self.gate_up_proj(x)

        if HAS_TRITON and gate_up.is_cuda and gate_up.dtype == torch.bfloat16:
            shape = gate_up.shape[:-1]
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
    """Replace components with fused versions. Returns counts."""
    if not HAS_TRITON:
        return {"mlp": 0, "rmsnorm": 0, "qkv": 0, "rope": 0}

    counts = {"mlp": 0, "rmsnorm": 0, "qkv": 0, "rope": 0}

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

        # Mark attention for fused RoPE
        attn._use_fused_rope = True
        counts["rope"] += 1

        # Mark layer for fused residual + RMSNorm
        layer._use_fused_residual_norm = True

    # Final RMSNorm
    if hasattr(model, 'norm'):
        model.norm = FusedRMSNorm(model.norm)
        counts["rmsnorm"] += 1

    return counts
