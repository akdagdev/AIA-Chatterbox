from dataclasses import dataclass
from typing import List, Union, Optional, Dict
from pathlib import Path
import os
import concurrent.futures
import threading
import time
import logging

_log = logging.getLogger(__name__)

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, EOS, SPEECH_VOCAB_SIZE, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

@dataclass
class SpeechRequest:
    text: str
    audio_prompt_path: Optional[str] = None
    language_id: Optional[str] = None
    conditionals: Optional[Conditionals] = None
    exaggeration: Optional[float] = None
    cfg_weight: Optional[float] = None


@dataclass
class T3PhaseResult:
    """Output of the T3 phase, input to the S3Gen phase.
    Allows pipelining: run S3Gen for batch N while T3 runs for batch N+1."""
    speech_tokens: torch.Tensor          # (batch_size, seq_len) — raw T3 output, PAD→EOS already applied
    item_ref_dicts: list                  # list[dict] — per-item S3Gen conditioning
    batch_size: int
    t_t3: float                           # T3 wall-clock seconds (for logging)


class ChatterboxMultilingualTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str,
        conds: Conditionals = None,
        s3gen_copies: list = None,  # Multiple S3Gen copies for parallel inference
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.s3gen_copies = s3gen_copies if s3gen_copies else [s3gen]  # Use copies for parallel streams
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self._batch_lock = threading.Lock()

        
        # Safely initialize watermarker
        # Safely initialize watermarker
        # if hasattr(perth, 'PerthImplicitWatermarker') and callable(perth.PerthImplicitWatermarker):
        #     self.watermarker = perth.PerthImplicitWatermarker()
        # else:
        import logging
        logging.warning("Watermarking disabled by user request.")
        # Simple dummy class to prevent errors later
        class DummyWatermarker:
            def apply_watermark(self, wav, sample_rate):
                return wav
        self.watermarker = DummyWatermarker()

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_local(cls, ckpt_dir, device, num_s3gen_copies=4) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)

        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", map_location=device, weights_only=True)
        )
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        # T3 mixed-precision: only the Llama backbone (95% of weights, bandwidth
        # bottleneck) runs in BF16. BF16 has the same dynamic range as FP32
        # (8 exponent bits) so attention scores and RMSNorm don't overflow/underflow
        # like FP16. Embeddings, conditioning, position embeddings, and the
        # speech_head projection stay FP32 for logit precision.
        # BF16 conversion is applied unconditionally on all CUDA GPUs including
        # Blackwell (sm_120) — bfloat16 is natively supported on all sm_80+.
        # Optional weight quantization (set T3_QUANTIZE env var):
        #   T3_QUANTIZE=fp8  — FP8 W8A8 dynamic (best: uses FP8 tensor cores, no dequant)
        #   T3_QUANTIZE=int8 — INT8 W8A8 dynamic (uses INT8 tensor cores, no dequant)
        #   T3_QUANTIZE=int4 — INT4 weight-only (4× bandwidth vs BF16, quality risk)
        # FP8/INT8 W8A8 run matmul directly in reduced precision — no dequantize
        # overhead. Weight-only modes (int4) dequantize to BF16 on-the-fly, which
        # adds overhead that can outweigh bandwidth savings on small models.
        # Requires torchao package. FP8 needs sm_89+ (Ada/Blackwell).
        if str(device).startswith("cuda"):
            cc_major, _ = torch.cuda.get_device_capability(device)
            t3.tfmr.to(torch.bfloat16)  # Llama backbone only

            # Fuse BEFORE quantization so fused weights (QKV, MLP gate_up) are
            # regular tensors that torchao can quantize. Reversing this order
            # makes torch.cat fail on AffineQuantizedTensor → fusions skipped →
            # quantization bandwidth gain cancelled by extra kernel launches.
            t3.init_patched_model()

            quant_mode = os.environ.get("T3_QUANTIZE", "").lower()
            if quant_mode in ("fp8", "int8", "int4"):
                try:
                    from torchao.quantization import quantize_
                    if quant_mode == "fp8":
                        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                    elif quant_mode == "int8":
                        from torchao.quantization import Int8DynamicActivationInt8WeightConfig
                    else:
                        from torchao.quantization import Int4WeightOnlyConfig
                except ImportError:
                    _log.warning("T3_QUANTIZE=%s requested but torchao not installed, staying BF16", quant_mode)
                    quant_mode = ""
                if quant_mode in ("fp8", "int8", "int4"):
                    try:
                        if quant_mode == "fp8":
                            config = Float8DynamicActivationFloat8WeightConfig()
                        elif quant_mode == "int8":
                            config = Int8DynamicActivationInt8WeightConfig()
                        else:
                            config = Int4WeightOnlyConfig()
                        quantize_(t3.patched_model.model, config)
                        _mode_desc = {"fp8": "FP8 W8A8", "int8": "INT8 W8A8", "int4": "INT4 weight-only"}
                        _log.info("T3 Llama backbone: %s + reduced compute (cc %d.x)", _mode_desc[quant_mode], cc_major)
                    except Exception as e:
                        _log.warning("T3_QUANTIZE=%s failed: %s — staying BF16", quant_mode, e)
                        _log.info("T3 Llama backbone running in BF16 (cc %d.x)", cc_major)
            else:
                _log.info("T3 Llama backbone running in BF16 (cc %d.x)", cc_major)

        # Note: torch.compile on the module only wraps forward() (training path).
        # Inference uses inference()/inference_batch() which bypass compiled forward.
        # Single-mode compilation is handled by standalone _generate_token_variants.

        # Load checkpoint once
        s3gen_state = torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)
        
        # Create S3Gen copies for parallel inference
        # First copy stays FP32 for embed_ref compatibility, others use FP16 for speed
        s3gen_copies = []
        for i in range(num_s3gen_copies):
            s3gen_copy = S3Gen()
            s3gen_copy.load_state_dict(s3gen_state)
            s3gen_copy.to(device)
            if i > 0:  # Only non-primary copies use FP16
                s3gen_copy.half()
                if str(device).startswith("cuda"):
                    # "default" mode (not "reduce-overhead"): no internal CUDA graph trees →
                    # thread-safe, respects torch.cuda.stream() context from worker threads →
                    # enables true GPU parallelism across 3 copies on separate streams.
                    # "reduce-overhead" binds graphs to the capture stream (default) and
                    # silently serializes all thread calls there regardless of stream context.
                    s3gen_copy.flow = torch.compile(s3gen_copy.flow, mode="default")
                    s3gen_copy.mel2wav = torch.compile(s3gen_copy.mel2wav, mode="default")
            s3gen_copy.eval()
            s3gen_copies.append(s3gen_copy)
        
        # Primary s3gen is the first copy (FP32 for embed_ref)
        s3gen = s3gen_copies[0]

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        # Pre-warm T3 prefill CUDA graphs for all bucket sizes so that the first
        # real inference call never triggers a slow capture mid-conversation.
        if str(device).startswith("cuda"):
            t3.init_patched_model()
            t3.warmup_prefill_graphs()

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds, s3gen_copies=s3gen_copies)

    @classmethod
    def from_pretrained(cls, device: torch.device, num_s3gen_copies=4) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device, num_s3gen_copies=num_s3gen_copies)
    
    def get_conditioning_for_prompt(self, wav_fpath, exaggeration=0.5):
        """
        Extract conditioning data (T3Cond and S3Gen ref_dict) from an audio file.
        Returns:
            t3_cond (T3Cond): Conditioning object for T3
            s3gen_ref_dict (dict): Reference dict for S3Gen
        """
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        t3_cond_prompt_tokens = None
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=ve_embed.dtype, device=self.device),
        ).to(device=self.device)
        
        return t3_cond, s3gen_ref_dict

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        t3_cond, s3gen_ref_dict = self.get_conditioning_for_prompt(wav_fpath, exaggeration)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def extract_voice_embedding(self, wav_fpath, exaggeration=0.5) -> Conditionals:
        """
        Extract a reusable voice embedding (Conditionals) from a reference audio file.
        This is a standalone utility — it does NOT modify self.conds.
        
        Args:
            wav_fpath: Path to the reference audio file (wav, mp3, etc.)
            exaggeration: Emotion/emphasis level (0.0 - 1.0)
            
        Returns:
            Conditionals: A portable voice embedding object that can be:
                - Passed to generate(conditionals=...) or SpeechRequest(conditionals=...)
                - Saved to disk via .save(path) and loaded via Conditionals.load(path)
                - Cached in memory for repeated use
        """
        t3_cond, s3gen_ref_dict = self.get_conditioning_for_prompt(wav_fpath, exaggeration)
        return Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        conditionals: Conditionals = None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # cache optimization params
        max_new_tokens=1000,
        max_cache_len=1500, # Affects the T3 speed, hence important
        # t3 sampling params
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        multi_step_n=4,
        t3_params={},
    ):
        """Full generate: T3 → S3Gen → watermark.  Thread-safe."""
        t3r = self.generate_t3(
            text=text, language_id=language_id,
            audio_prompt_path=audio_prompt_path, conditionals=conditionals,
            exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature,
            max_new_tokens=max_new_tokens, max_cache_len=max_cache_len,
            repetition_penalty=repetition_penalty, min_p=min_p, top_p=top_p,
            multi_step_n=multi_step_n, t3_params=t3_params,
        )
        return self.generate_s3gen(t3r)

    # ─── Single-item T3 phase ────────────────────────────────────────

    def generate_t3(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        conditionals: Conditionals = None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_cache_len=1500,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        multi_step_n=4,
        t3_params={},
    ) -> T3PhaseResult:
        """Text → speech tokens.  Thread-safe, runs on default CUDA stream."""
        import os
        if os.environ.get("T3_PROFILE") == "1":
            t3_params = {**t3_params, "profile_t3": True, "benchmark_t3": True}

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Resolve conditionals into a local variable (thread-safe)
        if conditionals is not None:
            conds = conditionals
        elif audio_prompt_path:
            t3_cond, s3gen_ref_dict = self.get_conditioning_for_prompt(audio_prompt_path, exaggeration)
            conds = Conditionals(t3_cond, s3gen_ref_dict)
        else:
            assert self.conds is not None, "Please specify `conditionals`, `audio_prompt_path`, or call `prepare_conditionals` first"
            conds = self.conds

        # Update exaggeration if needed (on local copy, not self)
        if exaggeration != conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = conds.t3
            conds = Conditionals(
                t3=T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
                ).to(device=self.device),
                gen=conds.gen,
            )

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            _is_cuda = self.device == "cuda" or (hasattr(self.device, "type") and self.device == "cuda")
            try:
                _is_cuda = next(self.t3.patched_model.parameters()).is_cuda
            except Exception:
                pass

            if _is_cuda:
                torch.cuda.synchronize()
            t_t3_start = time.perf_counter()
            speech_tokens = self.t3.inference(
                t3_cond=conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                multi_step_n=multi_step_n,
                **t3_params,
            )
            if _is_cuda:
                torch.cuda.synchronize()
            t_t3 = time.perf_counter() - t_t3_start

        return T3PhaseResult(
            speech_tokens=speech_tokens,
            item_ref_dicts=[conds.gen],
            batch_size=1,
            t_t3=t_t3,
        )

    # ─── Single-item S3Gen phase ─────────────────────────────────────

    def generate_s3gen(self, t3_result: T3PhaseResult) -> torch.Tensor:
        """Speech tokens → watermarked waveform [1, samples].
        Runs on a dedicated CUDA stream so it can truly overlap with T3 on
        the default stream (T3 is memory-bw bound, S3Gen is compute bound)."""
        speech_tokens = t3_result.speech_tokens
        ref_dict = t3_result.item_ref_dicts[0]
        t_t3 = t3_result.t_t3

        with torch.inference_mode():
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            s3gen_infer = self.s3gen_copies[1] if len(self.s3gen_copies) > 1 else self.s3gen

            _is_cuda = torch.cuda.is_available()

            # Lazy-init a dedicated CUDA stream for single-item S3Gen pipeline.
            # Reuses the first batch stream if available, otherwise creates one.
            if _is_cuda:
                if not hasattr(self, '_s3gen_single_stream'):
                    self._s3gen_single_stream = torch.cuda.Stream()

            t_s3_start = time.perf_counter()
            if _is_cuda:
                with torch.cuda.stream(self._s3gen_single_stream):
                    wav, _ = s3gen_infer.inference(
                        speech_tokens=speech_tokens,
                        ref_dict=ref_dict,
                    )
                self._s3gen_single_stream.synchronize()
            else:
                wav, _ = s3gen_infer.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=ref_dict,
                )
            t_s3 = time.perf_counter() - t_s3_start

            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)

        n_tokens = len(speech_tokens)
        audio_dur = n_tokens * 2 * 480 / self.sr
        from .models.s3gen.flow import S3GEN_EULER_STEPS
        _log.info(
            f"[SINGLE] T3={t_t3:.3f}s ({n_tokens} tokens, {n_tokens/t_t3:.0f} tok/s) | "
            f"S3Gen={t_s3:.3f}s (euler={S3GEN_EULER_STEPS}) | total={t_t3+t_s3:.3f}s | "
            f"audio={audio_dur:.2f}s | RTF={(t_t3+t_s3)/audio_dur:.3f}"
        )
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate_batch(
        self,
        texts: list,
        language_id: str,
        audio_prompt_path: Union[str, List[str]] = None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_cache_len=1500,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        t3_params={},
    ):
        """
        Batch TTS generation for multiple texts simultaneously.

        Args:
            texts: List of text strings OR List of SpeechRequest objects
            language_id: Language code (e.g., 'en', 'de', 'fr') - used if not in SpeechRequest
            audio_prompt_path: Optional path for voice cloning - used if not in SpeechRequest

        Returns:
            List of torch.Tensor, each containing generated audio waveform
        """
        if not self._batch_lock.acquire(blocking=False):
            raise RuntimeError(
                "generate_batch() is already running from another thread. "
                "This method is NOT thread-safe — use a single worker or queue."
            )
        try:
            return self._generate_batch_impl(
                texts=texts,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                t3_params=t3_params,
            )
        finally:
            self._batch_lock.release()

    def generate_batch_t3(self, **kwargs) -> T3PhaseResult:
        """Public T3-only phase.  Caller MUST hold _batch_lock or guarantee
        single-threaded access.  Returns T3PhaseResult for generate_batch_s3gen()."""
        return self._t3_phase(**kwargs)

    def generate_batch_s3gen(self, t3_result: T3PhaseResult) -> list:
        """Public S3Gen-only phase.  Can run concurrently with a T3 call
        on a different CUDA stream."""
        return self._s3gen_phase(t3_result)

    # ─── Phase 1: T3 (text → speech tokens) ────────────────────────────

    def _t3_phase(
        self,
        texts: list,
        language_id: str,
        audio_prompt_path: Union[str, List[str]] = None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_cache_len=1500,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        t3_params={},
    ) -> T3PhaseResult:
        """Text → speech tokens via T3.  Returns a T3PhaseResult that can be
        fed directly to _s3gen_phase().  Designed so that S3Gen for the
        *previous* batch can run concurrently on separate CUDA streams while
        this method occupies the default stream."""

        batch_size = len(texts)

        # Convert simple list of strings to SpeechRequest list for unified processing
        input_requests = []
        if isinstance(texts[0], str):
            if isinstance(audio_prompt_path, list):
                if len(audio_prompt_path) != batch_size:
                    raise ValueError(f"Number of audio prompts ({len(audio_prompt_path)}) must match batch size ({batch_size})")
                for i in range(batch_size):
                    input_requests.append(SpeechRequest(
                        text=texts[i],
                        audio_prompt_path=audio_prompt_path[i],
                        language_id=language_id
                    ))
            else:
                for t in texts:
                    input_requests.append(SpeechRequest(
                        text=t,
                        audio_prompt_path=audio_prompt_path,  # Could be str or None
                        language_id=language_id
                    ))
        else:
            # Assume it is already a list of SpeechRequests
            input_requests = texts

        # Prepare text list for tokenization
        processed_texts = [r.text for r in input_requests]

        # Determine language for each item (fallback to global arg if missing in request)
        req_lang = input_requests[0].language_id or language_id

        # Validate language_id
        if req_lang and req_lang.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{req_lang}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare conditioning
        batched_t3_cond = None
        item_ref_dicts = []

        has_custom_prompts = any(r.audio_prompt_path is not None for r in input_requests)

        if has_custom_prompts or any(r.conditionals is not None for r in input_requests):
            # Multi-speaker batching (or explicitly specified single speaker)
            t3_cond_list = []

            # Simple cache for path-based prompts within this batch
            prompt_cache = {}

            for req in input_requests:
                prompt_path = req.audio_prompt_path
                current_exaggeration = req.exaggeration if req.exaggeration is not None else exaggeration

                t3_c = None
                s3_d = None

                # Priority 1: Pre-computed conditionals
                if req.conditionals:
                    t3_c = req.conditionals.t3
                    s3_d = req.conditionals.gen

                # Priority 2: Audio path (compute or cache)
                elif prompt_path:
                    if prompt_path in prompt_cache:
                        t3_c, s3_d = prompt_cache[prompt_path]
                    else:
                        t3_c, s3_d = self.get_conditioning_for_prompt(prompt_path, exaggeration=1.0)
                        prompt_cache[prompt_path] = (t3_c, s3_d)

                # Priority 3: Fallback to global defaults
                else:
                    if hasattr(self, 'conds') and self.conds:
                        t3_c = self.conds.t3
                        s3_d = self.conds.gen
                    else:
                        raise ValueError("Some requests missing audio_prompt_path/conditionals and no default voice loaded.")

                # Override emotion_adv with per-item exaggeration
                t3_c = T3Cond(
                    speaker_emb=t3_c.speaker_emb,
                    cond_prompt_speech_tokens=t3_c.cond_prompt_speech_tokens,
                    emotion_adv=current_exaggeration * torch.ones(1, 1, 1, dtype=t3_c.speaker_emb.dtype, device=self.device),
                )
                t3_cond_list.append(t3_c)
                item_ref_dicts.append(s3_d)

            # Merge T3Conds
            speaker_emb = torch.cat([c.speaker_emb for c in t3_cond_list], dim=0)
            emotion_adv = torch.cat([c.emotion_adv for c in t3_cond_list], dim=0)

            prompt_tokens_list = [c.cond_prompt_speech_tokens for c in t3_cond_list if c.cond_prompt_speech_tokens is not None]

            cond_prompt_speech_tokens = None
            if prompt_tokens_list:
                 if len(prompt_tokens_list) == len(t3_cond_list):
                     max_len = max([t.shape[1] for t in prompt_tokens_list])
                     padded_prompts = []
                     for t in prompt_tokens_list:
                         if t.shape[1] < max_len:
                             pad_amount = max_len - t.shape[1]
                             t = F.pad(t, (0, pad_amount), value=0)
                         padded_prompts.append(t)
                     cond_prompt_speech_tokens = torch.cat(padded_prompts, dim=0)

            batched_t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_speech_tokens,
                emotion_adv=emotion_adv
            ).to(device=self.device)

        else:
            # No custom prompts — use global defaults (self.conds)
            if not hasattr(self, 'conds') or self.conds is None:
                 pass # Warning/Error handled elsewhere or relying on pre-load

            _cond: T3Cond = self.conds.t3

            exag_values = [
                r.exaggeration if r.exaggeration is not None else exaggeration
                for r in input_requests
            ]
            if len(set(exag_values)) > 1 or exag_values[0] != _cond.emotion_adv[0, 0, 0].item():
                emotion_adv = torch.tensor(exag_values, dtype=_cond.speaker_emb.dtype, device=self.device).view(-1, 1, 1)
                speaker_emb = _cond.speaker_emb.expand(batch_size, -1, -1).contiguous()
                cond_prompt = _cond.cond_prompt_speech_tokens
                if cond_prompt is not None and cond_prompt.shape[0] == 1:
                    cond_prompt = cond_prompt.expand(batch_size, -1).contiguous()
                batched_t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    cond_prompt_speech_tokens=cond_prompt,
                    emotion_adv=emotion_adv,
                ).to(device=self.device)
            else:
                batched_t3_cond = _cond

            item_ref_dicts = [self.conds.gen] * batch_size

        # Use the decided T3Cond
        t3_cond_to_use = batched_t3_cond if batched_t3_cond else self.conds.t3

        # Prepare language_id list
        language_ids = []
        for r in input_requests:
            lid = r.language_id or language_id
            if not lid:
                 raise ValueError(f"Language ID not specified for request '{r.text[:20]}...' and no global language_id provided")
            if lid.lower() not in SUPPORTED_LANGUAGES:
                 raise ValueError(f"Unsupported language_id '{lid}'")
            language_ids.append(lid.lower())

        # Batch tokenization
        normalized_texts = [punc_norm(t) for t in processed_texts]
        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token

        text_tokens, attention_mask = self.tokenizer.text_to_tokens_batch(
            normalized_texts,
            language_id=language_ids,
            sot_token=sot,
            eot_token=eot
        )
        text_tokens = text_tokens.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Build per-item cfg_weight tensor
        cfg_weights = [
            r.cfg_weight if r.cfg_weight is not None else cfg_weight
            for r in input_requests
        ]
        any_cfg = any(w > 0.0 for w in cfg_weights)
        cfg_weight_tensor = torch.tensor(cfg_weights, dtype=torch.float32, device=self.device).view(-1, 1)

        # CFG: duplicate tokens, attention_mask, and T3Cond only when any item uses CFG
        if any_cfg:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
            attention_mask = torch.cat([attention_mask, attention_mask], dim=0)

            if not (t3_cond_to_use.speaker_emb.shape[0] == 1 and batch_size > 1):
                cfg_t3_cond = T3Cond(
                    speaker_emb=torch.cat([t3_cond_to_use.speaker_emb, t3_cond_to_use.speaker_emb], dim=0),
                    cond_prompt_speech_tokens=torch.cat([t3_cond_to_use.cond_prompt_speech_tokens, t3_cond_to_use.cond_prompt_speech_tokens], dim=0) if t3_cond_to_use.cond_prompt_speech_tokens is not None else None,
                    emotion_adv=torch.cat([t3_cond_to_use.emotion_adv, t3_cond_to_use.emotion_adv], dim=0),
                ).to(device=self.device)
                t3_cond_to_use = cfg_t3_cond

        # ── T3 inference ──
        with torch.inference_mode():
            t_t3_start = time.perf_counter()
            speech_tokens = self.t3.inference_batch(
                t3_cond=t3_cond_to_use,
                text_tokens=text_tokens,
                text_attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight_tensor,
                use_cfg=any_cfg,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                **t3_params,
            )
            t_t3 = time.perf_counter() - t_t3_start

            # Replace PAD tokens with EOS — safety net for items that didn't generate EOS.
            PAD_TOKEN_ID = self.t3.hp.stop_speech_token + 1  # 6563
            speech_tokens = speech_tokens.clone()
            speech_tokens[speech_tokens == PAD_TOKEN_ID] = EOS

        return T3PhaseResult(
            speech_tokens=speech_tokens,
            item_ref_dicts=item_ref_dicts,
            batch_size=batch_size,
            t_t3=t_t3,
        )

    # ─── Phase 2: S3Gen (speech tokens → waveform) ───────────────────

    def _s3gen_phase(self, t3_result: T3PhaseResult) -> list:
        """Speech tokens → waveforms via S3Gen.  Runs on dedicated CUDA
        streams (one per FP16 copy) so it can overlap with a concurrent
        T3 call on the default stream."""

        batch_size = t3_result.batch_size
        speech_tokens = t3_result.speech_tokens
        item_ref_dicts = t3_result.item_ref_dicts

        s3gen_workers = self.s3gen_copies[1:] if len(self.s3gen_copies) > 1 else [self.s3gen]
        n_workers = len(s3gen_workers)

        MIN_SPEECH_TOKENS = 3

        # Pre-extract valid tokens for all items
        item_valid_tokens = []
        token_counts = []
        for i in range(batch_size):
            valid_tokens = drop_invalid_tokens(speech_tokens[i]).to(self.device)
            valid_tokens = valid_tokens[valid_tokens < SPEECH_VOCAB_SIZE]
            item_valid_tokens.append(valid_tokens)
            token_counts.append(len(valid_tokens) if len(valid_tokens) >= MIN_SPEECH_TOKENS else 0)

        use_cuda_streams = torch.cuda.is_available()

        # Lazy-init per-worker CUDA streams (one per FP16 copy, reused across batches)
        if use_cuda_streams:
            if not hasattr(self, '_s3gen_streams') or len(self._s3gen_streams) != n_workers:
                self._s3gen_streams = [torch.cuda.Stream() for _ in range(n_workers)]

        # Group items by worker (round-robin)
        worker_queues = [[] for _ in range(n_workers)]
        for i in range(batch_size):
            if token_counts[i] > 0:
                worker_queues[i % n_workers].append(i)

        raw_wavs = {}  # item_idx → tensor on GPU

        def run_worker(worker_idx, item_indices):
            worker = s3gen_workers[worker_idx]
            stream = self._s3gen_streams[worker_idx] if use_cuda_streams else None
            for idx in item_indices:
                tokens = item_valid_tokens[idx]
                if stream is not None:
                    with torch.cuda.stream(stream):
                        wav, _ = worker.inference(
                            speech_tokens=tokens,
                            ref_dict=item_ref_dicts[idx],
                        )
                        raw_wavs[idx] = wav.squeeze(0)[: len(tokens) * 2 * 480]
                else:
                    wav, _ = worker.inference(
                        speech_tokens=tokens,
                        ref_dict=item_ref_dicts[idx],
                    )
                    raw_wavs[idx] = wav.squeeze(0)[: len(tokens) * 2 * 480]
            if stream is not None:
                stream.synchronize()

        t_s3_start = time.perf_counter()
        with torch.inference_mode():
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futs = [
                    executor.submit(run_worker, w, worker_queues[w])
                    for w in range(n_workers)
                ]
                for f in futs:
                    f.result()  # re-raise any exceptions
        t_s3_total = time.perf_counter() - t_s3_start

        # Assemble results in original order, apply watermark (CPU-side, safe)
        results = []
        for i in range(batch_size):
            if token_counts[i] == 0:
                _log.warning(
                    f"Batch item {i} has only {len(item_valid_tokens[i])} valid speech tokens "
                    f"(min={MIN_SPEECH_TOKENS}), returning silence"
                )
                results.append(torch.zeros(1, int(0.1 * self.sr)))
                continue
            wav_numpy = raw_wavs[i].detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav_numpy, sample_rate=self.sr)
            results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))

        # Log timing
        total_tokens = sum(token_counts)
        total_audio = sum(n * 2 * 480 / self.sr for n in token_counts)
        t_t3 = t3_result.t_t3
        total = t_t3 + t_s3_total
        _log.info(
            f"[BATCH={batch_size}] T3={t_t3:.3f}s ({total_tokens} tokens, {total_tokens/t_t3:.0f} tok/s) | "
            f"S3Gen={t_s3_total:.3f}s (per-item avg {t_s3_total/max(batch_size,1):.3f}s) | "
            f"total={total:.3f}s | audio={total_audio:.2f}s | RTF={total/max(total_audio,1e-6):.3f}"
        )
        return results

    # ─── Combined (backward-compatible) ───────────────────────────────

    def _generate_batch_impl(
        self,
        texts: list,
        language_id: str,
        audio_prompt_path: Union[str, List[str]] = None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_cache_len=1500,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        t3_params={},
    ):
        if len(texts) == 0:
            return []
        t3_result = self._t3_phase(
            texts=texts,
            language_id=language_id,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            max_cache_len=max_cache_len,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            t3_params=t3_params,
        )
        return self._s3gen_phase(t3_result)
