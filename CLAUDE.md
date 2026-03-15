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
- Copies 1–3: FP16 — used for parallel waveform inference in batch mode

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
- `generate_batch()` uses `max_new_tokens=400` by default (vs 1000 for single). This is intentional for latency.
- Output sample rate is always `model.sr = 24000`.
- PerTh watermarking is always applied to generated audio.
- The custom Llama implementation in `src/chatterbox/models/t3/inference/custom_llama/modeling_llama.py` is a stripped-down fork of HuggingFace transformers' Llama — do not replace it with the upstream version.
- Supported languages: `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh`

## Known TODOs in Code

- `t3.py:235` — `torch.compile` synchronization not implemented
- `s3tokenizer/s3tokenizer.py` — FIXME: inherits `nn.Module` but processes wavs one-by-one
- `cond_enc.py:67` — CLAP embeddings not yet implemented
- `alignment_stream_analyzer.py` — monotonic masking may skip spaces
