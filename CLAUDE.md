# CLAUDE.md — AIA-Chatterbox

## Overview

Multilingual TTS library (23 languages). Two-stage pipeline:

1. **T3** — Llama-based text → discrete speech tokens (520M params)
2. **S3Gen** — Speech tokens → waveform via flow-matching diffusion + HiFiGAN vocoder

Installed as a package into AI-Server's venv: `pip install -e .`

---

## Package Structure

```
src/chatterbox/
├── mtl_tts.py                          # Primary API — ChatterboxMultilingualTTS
├── tts.py                              # English-only variant (not used by AI-Server)
├── vc.py                               # Voice conversion (not used by AI-Server)
└── models/
    ├── t3/
    │   ├── t3.py                       # T3 Llama model — inference() + inference_batch()
    │   ├── t3_cuda_graphs.py           # CUDA graph wrapper + bucketing
    │   └── modules/
    │       ├── cond_enc.py             # T3Cond dataclass + CondEncoder
    │       └── ...
    ├── s3gen/
    │   ├── s3gen.py                    # S3Gen vocoder — embed_ref() + inference()
    │   ├── flow.py                     # CausalMaskedDiffWithXvec (flow matching)
    │   └── hifigan.py                  # HiFTGenerator vocoder (mel → 24kHz wav)
    ├── voice_encoder/
    │   └── voice_encoder.py            # LSTM speaker embedding extractor (256D)
    └── tokenizers/
        └── tokenizer.py               # MTLTokenizer — language-aware BPE
```

---

## Core API (mtl_tts.py)

### Key Classes

```python
ChatterboxMultilingualTTS   # Main model — wraps T3 + S3Gen + VoiceEncoder
Conditionals                # Portable voice embedding (save/load/to(device))
SpeechRequest               # Batch generation request wrapper
T3Cond                      # Low-level T3 conditioning tensors (in cond_enc.py)
```

### ChatterboxMultilingualTTS

```python
# Loading
model = ChatterboxMultilingualTTS.from_pretrained(device)   # From HuggingFace
model = ChatterboxMultilingualTTS.from_local(ckpt_dir, device)

# Voice embedding
conds = model.extract_voice_embedding(wav_fpath, exaggeration=0.5) → Conditionals
model.prepare_conditionals(wav_fpath, exaggeration=0.5)             # Sets self.conds

# Synthesis
audio = model.generate(text, language_id, conditionals=None, audio_prompt_path=None,
                        exaggeration=0.5, cfg_weight=0.5, temperature=0.8,
                        max_new_tokens=1000) → Tensor [1, T]

audios = model.generate_batch(texts: list[SpeechRequest], language_id=None,
                               cfg_weight=0.5, max_new_tokens=400) → list[Tensor]
```

**Attribute:** `model.sr = 24000` (output sample rate)

### Conditionals

```python
@dataclass
class Conditionals:
    t3: T3Cond    # Conditioning for T3 (speaker_emb, speech prompt tokens, emotion_adv)
    gen: dict     # S3Gen ref_dict (prompt_token, prompt_feat, embedding, ...)

    def save(fpath: str)
    def load(fpath: str, map_location=None) → Conditionals
    def to(device: str) → Conditionals
```

Serialization via `torch.save`/`torch.load`. AI-Server uses `serialize_conditionals()` / `deserialize_conditionals()` to convert to/from `bytes` for gRPC transport.

### SpeechRequest

```python
@dataclass
class SpeechRequest:
    text: str
    audio_prompt_path: Optional[str] = None
    language_id: Optional[str] = None          # "en", "tr", "fr", etc.
    conditionals: Optional[Conditionals] = None
```

### Conditional Priority in generate()

```
1. conditionals= (explicit Conditionals object)  ← AI-Server always uses this
2. audio_prompt_path= (extracted on-the-fly)
3. self.conds (global fallback set via prepare_conditionals)
```

---

## T3 Model (models/t3/t3.py)

Llama-3 backbone (hidden=1536, speech vocab=8194, text vocab=2454 multilingual).

### Key Methods

```python
# Single generation — used by model.generate()
def inference(t3_cond: T3Cond, text_tokens, max_new_tokens, cfg_weight,
              temperature, min_p, top_p, repetition_penalty) → speech_tokens

# Batch generation — used by model.generate_batch()
def inference_batch(t3_cond: T3Cond, text_tokens, ...) → speech_tokens_list

# Compile static graph slots for CUDA capture
def init_patched_model(len_cond=34, text_tokens_size=153)
```

### T3Cond (models/t3/modules/cond_enc.py)

```python
@dataclass
class T3Cond:
    speaker_emb: Tensor                         # (B, 1, 256)  voice encoder output
    clap_emb: Optional[Tensor] = None           # unused
    cond_prompt_speech_tokens: Optional[Tensor] = None   # (B, L_prompt) reference tokens
    cond_prompt_speech_emb: Optional[Tensor] = None      # embeddings of prompt tokens
    emotion_adv: Optional[Tensor] = 0.5         # exaggeration scalar
```

---

## CUDA Graph Wrapper (models/t3/t3_cuda_graphs.py)

### How It Works

CUDA graphs capture the exact sequence of GPU operations. For replay, input tensors are updated in-place before each replay call (`.copy_()`). This gives ~10-20% inference speedup.

**Bucketing:** Token sequence length is rounded up to the nearest 250 (max 1500). One CUDA graph per bucket, shared across all calls with that bucket size.

```
get_next_bucket(seq_len, bucket_size=250) → 250 | 500 | 750 | 1000 | 1250 | 1500
```

### T3StepCUDAGraphWrapper (single generation)

```python
def __call__(speech_embedding_cache, output_logits, i_tensor, ...)
    # If bucket not captured yet: captures graph
    # Otherwise: copies inputs into static tensors, replays graph

def _capture_graph_for_bucket(bucket_key)   # One-time capture
def reset(bucket_key=None)                   # Clear captured graphs
def guard()                                  # Validate dtype consistency
```

### T3BatchStepCUDAGraphWrapper (batch generation)

Same as single-step but also tracks `finished_mask` for per-sequence early stopping.

### Critical Constraint

**CUDA graph capture must not happen concurrently with any other CUDA operation** (including model loading, other captures). A concurrent CUDA op during capture corrupts the graph, leading to `index_copy_(): index out of bounds` → `cudaErrorAssert` → process crash.

AI-Server pre-captures all 6 direct-path graphs at startup via `TTSService.warmup()`. Batch-path graphs (batch_size > 1) are captured lazily at runtime — this is safe because STT streaming sessions no longer load models on GPU.

---

## S3Gen Vocoder (models/s3gen/s3gen.py)

### Pipeline

```
speech_tokens → [Flow Matching Diffusion] → mel (25Hz, 100D) → [HiFiGAN] → wav (24kHz)
```

### Key Methods

```python
# Extract reference dict from audio (for conditioning)
embed_ref(wav: Tensor, sr: int, device) → ref_dict

# End-to-end inference
S3Token2Wav.inference(speech_tokens, ref_dict) → wav Tensor [1, T]

# Batch reference collation
collate_ref_dicts(ref_dicts: list[dict], device) → batched_ref_dict
```

### ref_dict Structure

```python
{
    "prompt_token":     list[Tensor],   # Discrete codes of reference audio
    "prompt_token_len": int,
    "prompt_feat":      list[Tensor],   # Mel spectrogram of reference
    "prompt_feat_len":  int,
    "embedding":        Tensor          # Speaker embedding vector
}
```

---

## Voice Encoder (models/voice_encoder/voice_encoder.py)

LSTM-based speaker embedding extractor.

```
audio (16kHz) → 128-dim mel → sliding 80-frame windows → LSTM(256) → proj → L2-norm → 256-dim embedding
```

```python
ve.embeds_from_wavs(wavs: list[Tensor], sr: int) → Tensor (B, 256)
```

---

## Tokenizer (models/tokenizers/tokenizer.py)

Language-aware BPE tokenizer.

```python
tokenizer.encode(text, language_id) → list[int]
tokenizer.text_to_tokens_batch(texts, language_id, sot_token, eot_token) → (tokens, mask)
```

**Language preprocessing:**
- `zh` → Cangjie glyph codes
- `ja` → Hiragana (pykakasi)
- `he` → Diacritics (ONNX model)
- `ko` → Jamo decomposition
- `fr` → Accent decomposition
- `ru` → Stress marks

**Special tokens:** `[START]=255`, `[STOP]=0`, `[PAD]`, `[UNK]`

---

## Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `exaggeration` | 0.5 | 0–1 | Emotion/emphasis intensity |
| `cfg_weight` | 0.5 | 0–1 | Reference voice fidelity |
| `temperature` | 0.8 | >0 | Token sampling diversity |
| `max_new_tokens` | 1000 | — | Single gen token limit |
| `max_new_tokens` (batch) | 400 | — | Batch gen token limit |
| `max_cache_len` | 1500 | — | KV-cache window size |
| `min_p` | 0.05 | 0–1 | Min prob filter |
| `top_p` | 1.0 | 0–1 | Nucleus sampling |
| `repetition_penalty` | 1.2 | ≥1 | Penalize repeated tokens |

---

## Supported Languages

```
ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh
```

---

## Thread Safety

- `generate()` — **thread-safe** (no shared mutable state during inference)
- `generate_batch()` — **NOT thread-safe** (must be called from single GPU worker)
- `extract_voice_embedding()` — **thread-safe** (does not modify `self.conds`)
- `prepare_conditionals()` — mutates `self.conds` — not thread-safe

---

## Package Setup

```bash
pip install -e .          # Install in editable mode
# or
uv pip install -e .       # In AI-Server's uv venv
```

Requires PyTorch ≥ 2.8, torchaudio ≥ 2.8. GPU target: CUDA 12.8 (RTX 5000).

Model weights auto-downloaded from `ResembleAI/chatterbox` on first `from_pretrained()` call.
