from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
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

        
        # Safely initialize watermarker
        if hasattr(perth, 'PerthImplicitWatermarker') and callable(perth.PerthImplicitWatermarker):
            self.watermarker = perth.PerthImplicitWatermarker()
        else:
            import logging
            logging.warning("PerthImplicitWatermarker not available or failed to load. Watermarking will be disabled.")
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
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxMultilingualTTS':
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
        
        # Optimize T3
        t3 = torch.compile(t3, mode="reduce-overhead")

        # Load checkpoint once
        s3gen_state = torch.load(ckpt_dir / "s3gen.pt", map_location=device, weights_only=True)
        
        # Create 4 S3Gen copies for parallel inference
        NUM_S3GEN_COPIES = 4
        s3gen_copies = []
        for i in range(NUM_S3GEN_COPIES):
            s3gen_copy = S3Gen()
            s3gen_copy.load_state_dict(s3gen_state)
            s3gen_copy.to(device).eval()
            s3gen_copies.append(s3gen_copy)
        
        # Primary s3gen is the first copy
        s3gen = s3gen_copies[0]

        tokenizer = MTLTokenizer(
            str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds, s3gen_copies=s3gen_copies)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ChatterboxMultilingualTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main", 
                allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt", "Cangjie5_TC.json"],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)
    
    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
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
            emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=ve_embed.dtype),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
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
        t3_params={},
    ):
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower() if language_id else None).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                **t3_params,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate_batch(
        self,
        texts: list,
        language_id: str,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_cache_len=1500,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
    ):
        """
        Batch TTS generation for multiple texts simultaneously.
        
        Args:
            texts: List of text strings to synthesize
            language_id: Language code (e.g., 'en', 'de', 'fr')
            audio_prompt_path: Optional path to reference audio for voice cloning
            
        Returns:
            List of torch.Tensor, each containing generated audio waveform
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        batch_size = len(texts)
        
        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
            ).to(device=self.device)

            # === Pipelined Micro-Batch Execution ===
            # We process the input batch in chunks (micro-batches).
            # While S3Gen is processing chunk N, T3 starts generating tokens for chunk N+1.
            # This overlaps the execution of T3 and S3Gen for better throughput.
            
            # Default micro-batch size (can be tuned)
            MB_SIZE = 8
            num_samples = len(texts)
            micro_batches = []
            
            # Prepare all inputs for T3 first (tokenization is fast enough)
            for i in range(0, num_samples, MB_SIZE):
                chunk_texts = texts[i : min(i + MB_SIZE, num_samples)]
                chunk_indices = range(i, min(i + MB_SIZE, num_samples))
                
                # Tokenize chunk
                chunk_normalized = [punc_norm(t) for t in chunk_texts]
                chunk_tokens, chunk_mask = self.tokenizer.text_to_tokens_batch(
                    chunk_normalized, 
                    language_id=language_id.lower() if language_id else None,
                    sot_token=sot,
                    eot_token=eot
                )
                chunk_tokens = chunk_tokens.to(self.device)
                
                # Replace PAD with EOT
                chunk_tokens = torch.where(
                    chunk_mask.to(self.device) == 0,
                    torch.tensor(eot, device=self.device),
                    chunk_tokens
                )
                
                # CFG expansion
                chunk_tokens = torch.cat([chunk_tokens, chunk_tokens], dim=0)
                
                micro_batches.append({
                    "indices": chunk_indices,
                    "text_tokens": chunk_tokens
                })

            wav_futures = [None] * num_samples
            
            # S3Gen streams for parallelism
            num_copies = len(self.s3gen_copies)
            s3gen_streams = [torch.cuda.Stream() for _ in range(num_copies)]

            # Helper to launch S3Gen async
            def launch_s3gen_async(mb_idx, speech_tokens_batch):
                mb_indices = micro_batches[mb_idx]["indices"]
                batch_len = len(mb_indices)
                
                for i in range(batch_len):
                    global_idx = mb_indices[i]
                    copy_idx = global_idx % num_copies
                    stream = s3gen_streams[copy_idx]
                    s3gen_copy = self.s3gen_copies[copy_idx]
                    
                    # Prepare tokens
                    item_tokens = drop_invalid_tokens(speech_tokens_batch[i])
                    item_tokens = item_tokens.to(self.device)
                    
                    with torch.cuda.stream(stream):
                        wav, _ = s3gen_copy.inference(
                            speech_tokens=item_tokens,
                            ref_dict=self.conds.gen,
                        )
                        # wav_futures stores the tensor on GPU. We should NOT sync here.
                        wav_futures[global_idx] = wav.squeeze(0)

            # Pipeline Loop
            # T3 is synchronous in the main stream (default stream)
            # S3Gen runs in separate streams
            
            for i, mb in enumerate(micro_batches):
                # 1. Run T3 for current micro-batch (Synchronizes implicitely with previous ops on default stream)
                # But S3Gen from previous MB runs in separate streams, so T3 runs in parallel with S3Gen(i-1)
                
                batch_speech_tokens = self.t3.inference_batch(
                    t3_cond=self.conds.t3,
                    text_tokens=mb["text_tokens"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                    max_cache_len=max_cache_len,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
                
                # 2. Launch S3Gen for this T3 output (Async)
                launch_s3gen_async(i, batch_speech_tokens)
            
            # Synchronize all streams at the very end
            torch.cuda.synchronize()
            
            # Post-process results (CPU-bound watermarking)
            results = []
            for i in range(num_samples):
                if wav_futures[i] is None:
                    print(f"FATAL: wav_futures[{i}] is None!")
                wav = wav_futures[i].detach().cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))
            
            # print(f"DEBUG: generate_batch returning {len(results)} items")
            return results
