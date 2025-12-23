"""
Streaming TTS API for Chatterbox.

This module provides a streaming interface that generates audio chunks
in real-time as tokens are produced.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens, SPEECH_VOCAB_SIZE
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .tts import Conditionals, punc_norm, REPO_ID


class ChatterboxTTSStream:
    """
    Streaming version of ChatterboxTTS that yields audio chunks.
    
    Usage:
        model = ChatterboxTTSStream.from_pretrained("cuda")
        for audio_chunk in model.generate_stream("Hello, world!"):
            # Process/play audio_chunk (torch.Tensor)
            pass
    """
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTSStream':
        ckpt_dir = Path(ckpt_dir)

        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(str(ckpt_dir / "tokenizer.json"))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTSStream':
        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=ve_embed.dtype),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate_stream(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        chunk_size: int = 50,
        max_new_tokens: int = 1000,
        max_cache_len: int = 1500,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        verbose: bool = True,
    ) -> Generator[Tuple[torch.Tensor, dict], None, None]:
        """
        Generate audio in a streaming fashion.
        
        Args:
            text: Text to synthesize.
            audio_prompt_path: Path to voice reference audio.
            chunk_size: Number of T3 tokens per audio chunk (~0.5s at 50 tokens).
            
        Yields:
            torch.Tensor: Audio chunks at 24kHz sample rate.
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please prepare_conditionals first or specify audio_prompt_path"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
            ).to(device=self.device)

        # Tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        # Streaming state
        cache_source = None
        is_first_chunk = True
        
        import time
        start_time = time.perf_counter()
        first_chunk_time = None
        total_audio_duration = 0

        # Stream tokens and convert to audio
        for token_chunk, is_final in self.t3.inference_stream(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            cfg_weight=cfg_weight,
            max_cache_len=max_cache_len,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            chunk_size=chunk_size,
        ):
            # Clean tokens - extract first batch, filter invalid
            tokens = token_chunk[0]
            tokens = tokens[tokens < SPEECH_VOCAB_SIZE]
            
            if len(tokens) == 0:
                continue

            # Convert tokens to audio
            audio_chunk, cache_source = self.s3gen.inference_chunk(
                speech_tokens=tokens,
                ref_dict=self.conds.gen,
                cache_source=cache_source,
                is_first_chunk=is_first_chunk,
                is_last_chunk=is_final,
            )
            
            # Metrics
            current_time = time.perf_counter()
            metrics = {}
            
            if is_first_chunk:
                first_chunk_time = current_time
                ttfa = first_chunk_time - start_time
                metrics["ttfa"] = ttfa
                if verbose:
                    print(f"Time to First Audio: {ttfa*1000:.2f}ms")

            is_first_chunk = False
            
            chunk_duration = audio_chunk.shape[-1] / self.sr
            total_audio_duration += chunk_duration
            
            if is_final:
                total_time = current_time - start_time
                metrics["rtf"] = total_time / total_audio_duration
                metrics["total_time"] = total_time
                metrics["total_audio"] = total_audio_duration
                if verbose:
                    print(f"RTF: {metrics['rtf']:.4f}")
                    print(f"Total Time: {total_time:.2f}s")
                    print(f"Total Audio: {total_audio_duration:.2f}s")

            # Apply watermark and yield
            wav = audio_chunk.squeeze(0).cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
            yield torch.from_numpy(watermarked_wav), metrics
