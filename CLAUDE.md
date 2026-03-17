# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AIA-Chatterbox is a multilingual TTS library (23 languages) forked from ResembleAI/chatterbox, customized for the white-ai ecosystem. It is installed as an editable package into AI-Server's venv. Two-stage pipeline:

1. **T3** — Llama-3 backbone (hidden=1536, 520M params): text → discrete speech tokens
2. **S3Gen** — Flow-matching diffusion + HiFiGAN vocoder: speech tokens → 24kHz waveform

Primary API: `ChatterboxMultilingualTTS` in `src/chatterbox/mtl_tts.py`. The English-only `ChatterboxTTS` (tts.py) and voice conversion `ChatterboxVC` (vc.py) are **not used by AI-Server**.

## Commands

```bash
# Install (editable mode — use uv in AI-Server context)
pip install -e .
uv pip install -e .

# Run tests
pytest                               # all tests
pytest tests/test_mtl_tts.py         # unit tests (no GPU needed)
pytest tests/test_tokenizer.py       # tokenizer tests (downloads vocab from HF)
pytest -m "not gpu"                  # skip GPU-requiring tests
python tests/test_batch_tokenizer.py # legacy manual runner

# Run examples
python example_tts.py
python example_chunked.py        # Sentence-by-sentence with RTF metrics
python benchmark_batch.py        # Batch performance benchmarking

# Interactive demos
python gradio_tts_app.py         # English TTS Gradio UI
python multilingual_app.py       # Multilingual Gradio UI
python gradio_vc_app.py          # Voice conversion Gradio UI
```

No linting, formatting, or CI/CD configuration exists in this repo.

## Architecture

### Inference Pipeline

```
text + language_id
    → MTLTokenizer (language-specific preprocessing + BPE)
    → T3 (Llama autoregressive, CUDA-graph-accelerated)
    → speech tokens
    → S3Gen (flow matching → mel → HiFiGAN)
    → 24kHz waveform
    → PerTh watermark (always applied)
```

### S3Gen Parallel Copies

`ChatterboxMultilingualTTS` creates **4 S3Gen copies** at load time:
- Copy 0: FP32 — used exclusively for `embed_ref()` (voice embedding extraction)
- Copies 1–3: FP16 — used for per-item sequential waveform inference in batch mode (round-robin)

**Batch S3Gen uses per-item sequential inference**, not tensor batching. UpsampleConformerEncoder's O(T²) attention causes OOM when batching (8 items × 400 tokens → ~655 MiB attention scores). Per-item with FP16 copies reduces this to ~41 MiB. Net batch latency impact: ~2s slower but avoids OOM entirely.

### CUDA Graph Acceleration

T3 uses CUDA graph capture with bucketed sequence lengths (250, 500, 750, 1000, 1250, 1500). One graph per bucket, reused across calls.

**Global lock:** `t3_cuda_graphs.CUDA_CAPTURE_LOCK` must be set by the host application (AI-Server) to prevent concurrent CUDA operations during graph capture. The lock is held only during capture (~10-50ms), not during replay.

**Critical constraint:** A concurrent CUDA operation during graph capture corrupts the graph → `index_copy_(): index out of bounds` → `cudaErrorAssert` → **process crash**. AI-Server pre-captures all 6 single-path graphs at startup via `TTSService.warmup()`. Batch-path graphs are captured lazily (safe because STT no longer runs on GPU concurrently).

### torch.compile

T3 model is compiled with `torch.compile(mode="reduce-overhead")` at load time. Note: `torch.compile` has a thread-safety TODO in `t3.py:235` — the compilation step itself is not synchronized.

### Token-to-Waveform Upsample Chain

The exact sample conversion from speech tokens to waveform samples:
- **Token → Mel**: `token_mel_ratio = 2` (UpsampleConformerEncoder)
- **Mel → Wav**: HiFiGAN upsample `[8, 5, 3]` = 120x, then ISTFT `hop_len = 4` → **480 samples/mel-frame**
- **Token → Wav**: `2 × 480 = 960 samples/token`

When calculating valid waveform length from token count: `wav_len = token_count * 2 * 480`

**Do NOT use** `token_count / 25.0 * 22050` — this uses the wrong sample rate (22050 vs 24000) and yields 882 samples/token, causing ~8% audio truncation.

### Conditioning Priority

When calling `generate()`, voice conditioning resolves in this order:
1. `conditionals=` parameter (explicit `Conditionals` object) — AI-Server always uses this
2. `audio_prompt_path=` (extracts embeddings on-the-fly each call)
3. `self.conds` (global fallback set via `prepare_conditionals()`)

### Tokenizer Language Preprocessing

The `MTLTokenizer` applies language-specific transforms before BPE:
- `zh` → Cangjie glyph codes (spacy-pkuseg)
- `ja` → Hiragana (pykakasi)
- `he` → Diacritics (ONNX model)
- `ko` → Jamo decomposition
- `fr` → Accent decomposition
- `ru` → Stress marks

## Thread Safety

- `generate()` — **thread-safe**
- `generate_batch()` — **NOT thread-safe** (must run from single GPU worker)
- `extract_voice_embedding()` — **thread-safe**
- `prepare_conditionals()` — mutates `self.conds` — **NOT thread-safe**

## Key Constraints

- **PyTorch ≥ 2.8, torchaudio ≥ 2.8** required. GPU target: CUDA 12.8 (RTX 5000). Dev: macOS Apple Silicon (MPS).
- Model weights auto-download from `ResembleAI/chatterbox` on first `from_pretrained()` call.
- `generate_batch()` default `max_new_tokens=1000`, but AI-Server passes 400 for latency. A dynamic cap of 4× the longest text's token count further limits iterations.
- Output sample rate is always `model.sr = 24000`.
- PerTh watermarking is always applied to generated audio.
- The custom Llama implementation in `src/chatterbox/models/t3/inference/custom_llama/modeling_llama.py` is a stripped-down fork of HuggingFace transformers' Llama — do not replace it with the upstream version.
- Supported languages: `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`

### Batch Text Padding — Zero-Masking (not PAD→EOT)

When batch items have different text lengths, the tokenizer pads shorter items. Instead of replacing PAD tokens with EOT (which creates multiple EOT tokens the model never saw during training), `generate_batch()` passes the `text_attention_mask` through to `T3.prepare_input_embeds()`, which zeros out padded positions' embeddings (content + positional) after all other processing.

**Why this works:** Llama uses `bias=False` throughout (attention projections, MLP). Zero embeddings produce zero K/V vectors: `K[pad] = W_k @ 0 = 0`, `V[pad] = W_v @ 0 = 0`. Regardless of attention weights, `weight * 0 = 0` — padded positions contribute nothing to the output. The model effectively "doesn't see" them.

Additionally, a **symmetric attention mask** is passed to Llama's forward call: the **same** mask is used for both conditioned and unconditioned CFG halves. This makes padded positions invisible in attention (not just in value), so BOS immediately "follows" real text from the model's perspective — matching per-item single-mode behavior. The mask is the tokenizer's original `text_attention_mask` duplicated for CFG; no special unconditioned mask is constructed.

**Do NOT** use an asymmetric attention mask (e.g., `zeros_like` or `ones_like` for the unconditioned half) — this breaks CFG by making the conditioned and unconditioned halves see different effective sequence structures.

**Do NOT** omit the attention mask entirely — without it, the zero-embedding gap between real text end and BOS confuses the model, causing it to never generate EOS (max-length random output).

**Do NOT** replace PAD tokens with EOT — this creates `[SOT, t1, t2, EOT, EOT, EOT]` sequences that destabilize T3's attention patterns and cause stuttering/hallucinations.

**Batch uses CUDA graphs** via `T3BatchStepCUDAGraphWrapper`. Root cause of the historic "0 valid tokens" bug: `torch.multinomial` inside `torch.cuda.graph()` capture context returns token 0 regardless of logits. Fix: `__call__` saves `original_gen_ids` before calling `_capture_graph_for_bucket`, then immediately runs one eager pass with `original_gen_ids` after capture to get the real first token. All subsequent steps use graph replay normally. Batch CUDA graphs run at ~130 it/s vs ~66-80 it/s in eager mode.

**gen_max_position optimization** — the generation loop uses `gen_max_position = get_next_bucket(seq_len + max_new_tokens)` (the smallest bucket covering the actual generation range) instead of the full cache size. KV cache reads scale linearly with max_position, so using bucket 500 instead of 1500 gives ~3× fewer reads for typical short texts, yielding meaningful speedup within eager mode.

**4D attention mask in generation loop** — the 2D attention mask is used for the initial forward pass (prefill), then converted to a 4D mask `(batch, 1, 1, gen_max_position)` for the generation loop. Updated per-step by unmasking one new position. This is necessary for correctness (not for CUDA graph compatibility) — passing a 2D mask during generation would re-trigger `_prepare_4d_causal_attention_mask_with_cache_position` each step with dynamic shape allocations.


### Batch vs Single EOS Handling

`inference_batch()` checks every generated token for EOS immediately — no guesstimate guard. When an item generates EOS, it is marked finished and all subsequent tokens are force-set to EOS. `drop_invalid_tokens` truncates at the first EOS regardless, so delaying detection only wastes loop iterations without affecting audio quality. A dynamic `max_new_tokens` cap (4× longest text tokens, minimum 100) prevents runaway loops. The loop exits as soon as all batch items have generated at least one EOS token.

## Known TODOs in Code

- `t3.py:235` — `torch.compile` synchronization not implemented
- `s3tokenizer/s3tokenizer.py` — FIXME: inherits `nn.Module` but processes wavs one-by-one
- `cond_enc.py:67` — CLAP embeddings not yet implemented
- `alignment_stream_analyzer.py` — monotonic masking may skip spaces
