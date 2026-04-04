"""
GPU performance diagnostic for T3 inference bottleneck.
Run on both GPUs and compare output.

Usage: python diagnose_gpu.py
"""
import os
import sys
import torch
import time


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def diagnose():
    device = torch.device("cuda")

    # ── 1. GPU & Software Info ──
    section("1. GPU & SOFTWARE INFO")
    print(f"GPU: {torch.cuda.get_device_name()}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute Capability: sm_{cap[0]}{cap[1]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    try:
        import triton
        print(f"Triton: {triton.__version__}")
        HAS_TRITON = True
    except ImportError:
        print("Triton: NOT INSTALLED")
        HAS_TRITON = False

    print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
    print(f"Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
    print(f"T3_QUANTIZE: {os.environ.get('T3_QUANTIZE', 'NOT SET')}")
    print(f"T3_NO_FUSED: {os.environ.get('T3_NO_FUSED', 'NOT SET')}")

    # ── 2. Memory Bandwidth Test ──
    section("2. MEMORY BANDWIDTH TEST (raw)")
    # Large tensor copy — measures achievable bandwidth
    size_gb = 2.0
    n_elements = int(size_gb * 1e9 / 4)  # float32
    a = torch.randn(n_elements, device=device, dtype=torch.float32)
    b = torch.empty_like(a)

    # Warmup
    for _ in range(3):
        b.copy_(a)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    n_iters = 20
    start.record()
    for _ in range(n_iters):
        b.copy_(a)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    bytes_moved = 2 * n_elements * 4 * n_iters  # read + write
    bw_gb_s = bytes_moved / (elapsed_ms / 1000) / 1e9
    print(f"Measured bandwidth: {bw_gb_s:.1f} GB/s (2GB copy × {n_iters})")
    del a, b

    # ── 3. Triton Kernel Compilation Test ──
    section("3. TRITON KERNEL TEST")
    if HAS_TRITON:
        try:
            from chatterbox.models.t3.inference.custom_llama.fused_mlp import (
                FusedRMSNorm, fused_apply_rotary_pos_emb, fused_residual_rmsnorm,
                HAS_TRITON as FUSED_HAS_TRITON,
            )
            print(f"fused_mlp.HAS_TRITON = {FUSED_HAS_TRITON}")

            if FUSED_HAS_TRITON:
                # Test RMSNorm Triton kernel
                hidden_size = 1536
                x = torch.randn(2, 1, hidden_size, device=device, dtype=torch.bfloat16)

                class DummyNorm:
                    weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
                    variance_epsilon = 1e-5

                fused_norm = FusedRMSNorm(DummyNorm())

                # Warmup — this triggers Triton JIT compilation
                try:
                    for _ in range(5):
                        _ = fused_norm(x)
                    torch.cuda.synchronize()
                    print("RMSNorm Triton kernel: OK (compiled & ran)")
                except Exception as e:
                    print(f"RMSNorm Triton kernel: FAILED — {e}")

                # Benchmark Triton vs eager
                n_iters = 1000
                torch.cuda.synchronize()
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(n_iters):
                    _ = fused_norm(x)
                e.record()
                torch.cuda.synchronize()
                triton_ms = s.elapsed_time(e) / n_iters
                print(f"RMSNorm Triton: {triton_ms*1000:.1f} µs/call")

                # Eager fallback
                def eager_rmsnorm(x, weight, eps):
                    x32 = x.float()
                    var = x32.pow(2).mean(-1, keepdim=True)
                    return weight * (x32 * torch.rsqrt(var + eps)).to(x.dtype)

                w = DummyNorm.weight
                eps = DummyNorm.variance_epsilon
                for _ in range(5):
                    _ = eager_rmsnorm(x, w, eps)
                torch.cuda.synchronize()
                s2 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                s2.record()
                for _ in range(n_iters):
                    _ = eager_rmsnorm(x, w, eps)
                e2.record()
                torch.cuda.synchronize()
                eager_ms = s2.elapsed_time(e2) / n_iters
                print(f"RMSNorm eager:  {eager_ms*1000:.1f} µs/call")
                print(f"Triton speedup: {eager_ms/triton_ms:.2f}x")
        except Exception as ex:
            print(f"Triton kernel import/test failed: {ex}")
    else:
        print("SKIPPED — Triton not available")

    # ── 4. SDPA Backend Detection ──
    section("4. SDPA BACKEND TEST")
    # Simulate single-token attention (like T3 decode step)
    # batch=2 (CFG), heads=12, q_len=1, kv_len=250, head_dim=128
    batch, heads, kv_len, head_dim = 2, 12, 250, 128
    q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(batch, heads, kv_len, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch, heads, kv_len, head_dim, device=device, dtype=torch.bfloat16)

    # Check which backend is used
    backends = {
        "flash": torch.backends.cuda.flash_sdp_enabled,
        "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled,
        "math": torch.backends.cuda.math_sdp_enabled,
    }
    for name, check in backends.items():
        print(f"  {name}: {'enabled' if check() else 'DISABLED'}")

    # Benchmark SDPA
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    n_iters = 1000
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iters):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    e.record()
    torch.cuda.synchronize()
    sdpa_ms = s.elapsed_time(e) / n_iters
    print(f"SDPA (batch=2, heads=12, q=1, kv={kv_len}): {sdpa_ms*1000:.1f} µs/call")

    del q, k, v

    # ── 5. Small matmul benchmark (simulates T3 decode) ──
    section("5. SMALL MATMUL BENCHMARK (T3 decode-like)")
    # These mirror the actual T3 per-token matmuls
    configs = [
        ("QKV fused",   2, 1, 1536, 4608),
        ("Output proj", 2, 1, 1536, 1536),
        ("Gate+Up",     2, 1, 1536, 8192),
        ("Down proj",   2, 1, 4096, 1536),
        ("Speech head", 2, 1, 1536, 6561),  # speech vocab size
    ]
    for name, B, S, M, N in configs:
        x = torch.randn(B, S, M, device=device, dtype=torch.bfloat16)
        w = torch.randn(N, M, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.linear(x, w)
        torch.cuda.synchronize()

        n_iters = 1000
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            _ = torch.nn.functional.linear(x, w)
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e) / n_iters

        # Theoretical: 2*B*S*M*N FLOPs, read B*S*M + M*N bytes (bf16)
        bytes_read = (B * S * M + M * N) * 2  # bf16
        print(f"  {name:15s} ({B}×{S}×{M})×({N}×{M}): {ms*1000:.1f} µs  "
              f"(weight={M*N*2/1e6:.1f}MB, eff_bw={bytes_read/(ms/1000)/1e9:.0f} GB/s)")
        del x, w

    # ── 6. Full model single-step benchmark (eager, no CUDA graphs) ──
    section("6. FULL MODEL EAGER FORWARD PASS")
    try:
        from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
        from chatterbox.models.t3.inference.custom_llama.modeling_llama import LlamaConfig

        print("Loading model for eager benchmark...")
        # We need the actual model — try loading
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        model.t3.init_patched_model()

        patched = model.t3.patched_model
        config = patched.config
        hidden = config.hidden_size

        # Create minimal inputs
        from transformers.cache_utils import StaticCache
        cache = StaticCache(
            config=config,
            max_batch_size=2,
            max_cache_len=500,
            device=device,
            dtype=torch.bfloat16,
        )

        # Prefill with dummy data (so cache is populated)
        dummy_embeds = torch.randn(2, 50, hidden, device=device, dtype=torch.bfloat16)
        cache_pos = torch.arange(50, device=device, dtype=torch.long)
        with torch.inference_mode():
            _ = patched(
                inputs_embeds=dummy_embeds,
                past_key_values=cache,
                cache_position=cache_pos,
            )

        # Single-token forward (like decode step)
        token_embed = torch.randn(2, 1, hidden, device=device, dtype=torch.bfloat16)

        # Warmup
        for _ in range(10):
            cache_pos_single = cache.get_seq_length().unsqueeze(0)
            with torch.inference_mode():
                _ = patched(
                    inputs_embeds=token_embed,
                    past_key_values=cache,
                    cache_position=cache_pos_single,
                    max_position=250,
                )

        torch.cuda.synchronize()

        # Benchmark
        n_iters = 100
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n_iters):
            cache_pos_single = cache.get_seq_length().unsqueeze(0)
            with torch.inference_mode():
                _ = patched(
                    inputs_embeds=token_embed,
                    past_key_values=cache,
                    cache_position=cache_pos_single,
                    max_position=250,
                )
        e.record()
        torch.cuda.synchronize()
        ms_per_step = s.elapsed_time(e) / n_iters
        print(f"Eager forward (single token, max_pos=250): {ms_per_step:.3f} ms/step")
        print(f"Equivalent tok/s: {1000/ms_per_step:.0f}")

        # Compare with CUDA graph replay
        section("7. CUDA GRAPH REPLAY BENCHMARK")
        from chatterbox.models.t3.t3_cuda_graphs import T3StepCUDAGraphWrapper
        from chatterbox.models.t3.t3 import generate_t3_token

        # Reset cache for graph test
        cache.reset()
        dummy_embeds2 = torch.randn(2, 50, hidden, device=device, dtype=torch.bfloat16)
        cache_pos2 = torch.arange(50, device=device, dtype=torch.long)
        with torch.inference_mode():
            output_logits = patched(
                inputs_embeds=dummy_embeds2,
                past_key_values=cache,
                cache_position=cache_pos2,
            )
        print(f"CUDA graph test: prefill done, cache_len={cache.get_seq_length().item()}")
        print("(Graph capture & replay benchmarking requires full inference setup — skipped)")

    except Exception as ex:
        print(f"Model benchmark failed: {ex}")
        import traceback
        traceback.print_exc()

    section("DONE")
    print("\nCompare these results between RTX PRO 6000 and RTX 4090.")
    print("Key things to look for:")
    print("  - Section 2: bandwidth should be ~1.8TB/s (PRO 6000) vs ~1.0TB/s (4090)")
    print("  - Section 3: Triton kernel FAILED → fused kernels not working")
    print("  - Section 4: SDPA timing much higher → Flash Attention not working")
    print("  - Section 5: matmul eff_bw much lower → cuBLAS not optimized")
    print("  - Section 6: eager forward time → baseline without CUDA graph overhead")


if __name__ == "__main__":
    diagnose()
