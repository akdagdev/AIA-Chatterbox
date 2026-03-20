"""
Mixed pipeline test: direct generates, varied-size batches, and more directs.
Verifies that batch CUDA graphs don't break across calls and that
direct/batch interleaving works correctly.

Sequence:
  2 direct → 5 batches (8,8,7,4,3) → 3 direct → 1 batch (8) → 2 direct
All audio concatenated with 0.3s silence gaps → test_mixed_output.wav
"""

import time
import torch
import soundfile as sf
from pathlib import Path

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SpeechRequest, Conditionals

# ── Texts ────────────────────────────────────────────────────────────────────

TEXTS_EN = [
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Artificial intelligence is transforming every industry around the world today.",
    "She sells seashells by the seashore on a warm sunny afternoon.",
    "The weather forecast predicts light rain for the next three days.",
    "Call me back.",
    "Please remember to pick up some groceries on your way home today.",
    "Technology continues to evolve at an unprecedented pace every single year.",
    "The concert was absolutely fantastic and the crowd loved every moment of it.",
    "I would like to schedule a quick meeting for tomorrow morning at nine.",
    "Mountains covered in fresh snow create a breathtaking landscape in winter.",
    "The new restaurant downtown serves the most delicious homemade pasta in town, and I highly recommend their signature carbonara with freshly grated parmesan cheese on top.",
    "Reading books before bed helps you relax and fall asleep much faster.",
    "Sure, no problem.",
    "Our team has been working really hard to deliver this project on time.",
    "The sunset painted the sky in beautiful warm shades of orange and pink.",
    "Learning a new language opens doors to different cultures and interesting people.",
    "The train departs from platform three at exactly half past six this evening.",
]

TEXTS_TR = [
    "Merhaba, bugün hava çok güzel ve güneşli bir gün olacak gibi görünüyor.",
    "Yapay zeka teknolojisi her geçen gün biraz daha da gelişiyor.",
    "Tamam, geliyorum.",
    "İstanbul'un tarihi yarımadası dünyanın en güzel ve etkileyici yerlerinden biridir.",
    "Toplantı yarın sabah saat dokuzda başlayacak, lütfen zamanında hazır olun.",
    "Bu restoran şehirdeki en iyi köfteci olarak biliniyor herkes tarafından, özellikle de hafta sonu öğle yemeklerinde masaları bulmak neredeyse imkansız oluyor.",
]

ALL_TEXTS = [(t, "en") for t in TEXTS_EN] + [(t, "tr") for t in TEXTS_TR]


def save_audio(tensor: torch.Tensor, path: str, sr: int = 24000):
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    sf.write(path, tensor.detach().cpu().numpy().T, sr)


def make_silence(sr: int, duration: float = 0.3) -> torch.Tensor:
    return torch.zeros(1, int(sr * duration))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading model...")
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    sr = model.sr

    # ── Voice embedding ──────────────────────────────────────────────────
    ref_path = "referencebill.mp3"
    if not Path(ref_path).exists():
        ref_path = None
        print("No reference audio found, using default voice.")

    conds = None
    if ref_path:
        print(f"Extracting voice embedding from {ref_path}...")
        conds = model.extract_voice_embedding(ref_path, exaggeration=0.5)

    # ── Helpers ───────────────────────────────────────────────────────────
    all_audio: list[torch.Tensor] = []
    step_counter = [0]

    def direct_generate(text: str, lang: str):
        step_counter[0] += 1
        idx = step_counter[0]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wav = model.generate(
                text=text,
                language_id=lang,
                conditionals=conds,
                max_new_tokens=1000,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        dur = wav.shape[-1] / sr
        rtf = elapsed / dur if dur > 0 else 0
        print(f"  [direct #{idx}] {elapsed:.2f}s | {dur:.1f}s audio | RTF={rtf:.3f} | \"{text[:50]}...\"")
        all_audio.append(wav)

    def batch_generate(items: list[tuple[str, str]]):
        step_counter[0] += 1
        idx = step_counter[0]
        requests = [
            SpeechRequest(text=t, language_id=l, conditionals=conds)
            for t, l in items
        ]
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            wavs = model.generate_batch(
                texts=requests,
                language_id=None,
                max_new_tokens=400,
                cfg_weight=0.5,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_dur = sum(w.shape[-1] / sr for w in wavs)
        rtf = elapsed / total_dur if total_dur > 0 else 0
        print(f"  [batch  #{idx}] {elapsed:.2f}s | {len(items)} items | {total_dur:.1f}s audio | RTF={rtf:.3f}")
        all_audio.extend(wavs)

    # ── Sequence ─────────────────────────────────────────────────────────
    ti = 0  # text index into ALL_TEXTS

    # Phase 1: 2 direct generates
    print("\n── Phase 1: 2 direct generates ──")
    for _ in range(2):
        t, l = ALL_TEXTS[ti % len(ALL_TEXTS)]
        direct_generate(t, l)
        ti += 1

    # Phase 2: 5 batches (8, 8, 7, 4, 3)
    print("\n── Phase 2: 5 batches (8, 8, 7, 4, 3) ──")
    for batch_sz in [8, 8, 7, 4, 3]:
        items = []
        for _ in range(batch_sz):
            items.append(ALL_TEXTS[ti % len(ALL_TEXTS)])
            ti += 1
        batch_generate(items)

    # Phase 3: 3 direct generates
    print("\n── Phase 3: 3 direct generates ──")
    for _ in range(3):
        t, l = ALL_TEXTS[ti % len(ALL_TEXTS)]
        direct_generate(t, l)
        ti += 1

    # Phase 4: 1 batch (8)
    print("\n── Phase 4: 1 batch (8) ──")
    items = []
    for _ in range(8):
        items.append(ALL_TEXTS[ti % len(ALL_TEXTS)])
        ti += 1
    batch_generate(items)

    # Phase 5: 2 direct generates
    print("\n── Phase 5: 2 direct generates ──")
    for _ in range(2):
        t, l = ALL_TEXTS[ti % len(ALL_TEXTS)]
        direct_generate(t, l)
        ti += 1

    # ── Concatenate all audio ────────────────────────────────────────────
    print(f"\nConcatenating {len(all_audio)} audio segments...")
    silence = make_silence(sr, 0.3)
    parts = []
    for wav in all_audio:
        parts.append(wav.cpu() if wav.dim() == 2 else wav.cpu().unsqueeze(0))
        parts.append(silence)
    if parts:
        parts.pop()  # remove trailing silence

    combined = torch.cat(parts, dim=1)
    out_path = "test_mixed_output.wav"
    save_audio(combined, out_path, sr)

    total_dur = combined.shape[-1] / sr
    print(f"\nDone! Saved {total_dur:.1f}s of audio to {out_path}")
    print(f"Total segments: {len(all_audio)} (7 direct + {len(all_audio) - 7} batch items)")


if __name__ == "__main__":
    main()
