from dataclasses import dataclass
from typing import List, Union, Optional, Dict
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

@dataclass
@dataclass
class SpeechRequest:
    text: str
    audio_prompt_path: Optional[str] = None
    language_id: Optional[str] = None
    conditionals: Optional[Conditionals] = None


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
        # First copy stays FP32 for embed_ref compatibility, others use FP16 for speed
        NUM_S3GEN_COPIES = 4
        s3gen_copies = []
        for i in range(NUM_S3GEN_COPIES):
            s3gen_copy = S3Gen()
            s3gen_copy.load_state_dict(s3gen_state)
            s3gen_copy.to(device)
            if i > 0:  # Only non-primary copies use FP16
                s3gen_copy.half()
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
        audio_prompt_path: Union[str, List[str]] = None,
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
            texts: List of text strings OR List of SpeechRequest objects
            language_id: Language code (e.g., 'en', 'de', 'fr') - used if not in SpeechRequest
            audio_prompt_path: Optional path for voice cloning - used if not in SpeechRequest
            
        Returns:
            List of torch.Tensor, each containing generated audio waveform
        """
        batch_size = len(texts)
        if batch_size == 0:
            return []

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
        # Note: current tokenizer `text_to_tokens_batch` accepts a single language_id for the whole batch
        # If mixed languages are needed, tokenizer needs update. For now, we enforce single language or assume primary language.
        # Checking if all requests have same language or if we use global
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

        # Check if we have mixed prompts or single prompt
        # We need to build conditioning for EACH item to be safe, or optimize if all same.
        # Optimization: check if all prompt paths are None or same string?
        # For simplicity and correctness with the new Struct design, let's process each one.
        
        has_custom_prompts = any(r.audio_prompt_path is not None for r in input_requests)
        
        if has_custom_prompts or any(r.conditionals is not None for r in input_requests):
            # Multi-speaker batching (or explicitly specified single speaker)
            t3_cond_list = []
            
            # Simple cache for path-based prompts within this batch
            prompt_cache = {} 

            for req in input_requests:
                prompt_path = req.audio_prompt_path
                current_exaggeration = exaggeration # Support per-request exaggeration? For now global.
                
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
                        t3_c, s3_d = self.get_conditioning_for_prompt(prompt_path, current_exaggeration)
                        prompt_cache[prompt_path] = (t3_c, s3_d)
                
                # Priority 3: Fallback to global defaults
                else:
                    if hasattr(self, 'conds') and self.conds:
                        t3_c = self.conds.t3
                        s3_d = self.conds.gen
                    else:
                        raise ValueError("Some requests missing audio_prompt_path/conditionals and no default voice loaded.")

                t3_cond_list.append(t3_c)
                item_ref_dicts.append(s3_d)
            
            # Merge T3Conds logic...
            # Stack speaker embeddings
            speaker_emb = torch.cat([c.speaker_emb for c in t3_cond_list], dim=0)
            emotion_adv = torch.cat([c.emotion_adv for c in t3_cond_list], dim=0)
            
            # Stack prompt tokens
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
            # No custom prompts in requests, use global defaults
            # ... (existing single speaker logic) ...
            
            # Use global audio_prompt_path if provided (but we successfully parsed it into requests above)
            # If all requests have None prompt, but global arg was passed, input_requests has it.
            # So `has_custom_prompts` would be true if global arg was passed.
            # Thus we land here only if NO prompts provided anywhere.
            
            # Verify self.conds is set
            if not hasattr(self, 'conds') or self.conds is None:
                 pass # Warning/Error handled elsewhere or relying on pre-load

            batched_t3_cond = self.conds.t3
            
            # Update exaggeration if needed (global only)
            if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
                _cond: T3Cond = self.conds.t3
                # Optimization: Check if we need to clone.
                # If we modify self.conds in place it affects future calls.
                # Code below creates new T3Cond, updates self.conds.t3
                self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
                ).to(device=self.device)
                batched_t3_cond = self.conds.t3 # Update reference

            item_ref_dicts = [self.conds.gen] * batch_size

        # Use the decided T3Cond
        t3_cond_to_use = batched_t3_cond if batched_t3_cond else self.conds.t3

        # Prepare language_id list
        # If input_requests (SpeechRequest objects) have language_id, use it.
        # Fallback to global language_id.
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
        
        # Replace PAD tokens with EOT to prevent model confusion
        # PAD token is typically 0 or vocab["[PAD]"]
        pad_token_id = 0  # Default PAD token
        text_tokens = torch.where(
            attention_mask.to(self.device) == 0,
            torch.tensor(eot, device=self.device),
            text_tokens
        )

        # CFG: duplicate tokens and T3Cond only when CFG is enabled
        if cfg_weight > 0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
            
            # Only duplicate conds if we are NOT relying on broadcasting
            # If cond size is 1 and batch > 1, we leave it as 1 to broadcast to (2*batch) inside T3
            if not (t3_cond_to_use.speaker_emb.shape[0] == 1 and batch_size > 1):
                cfg_t3_cond = T3Cond(
                    speaker_emb=torch.cat([t3_cond_to_use.speaker_emb, t3_cond_to_use.speaker_emb], dim=0),
                    cond_prompt_speech_tokens=torch.cat([t3_cond_to_use.cond_prompt_speech_tokens, t3_cond_to_use.cond_prompt_speech_tokens], dim=0) if t3_cond_to_use.cond_prompt_speech_tokens is not None else None,
                    emotion_adv=torch.cat([t3_cond_to_use.emotion_adv, t3_cond_to_use.emotion_adv], dim=0),
                ).to(device=self.device)
                t3_cond_to_use = cfg_t3_cond

        with torch.inference_mode():
            # T3 batch inference
            speech_tokens = self.t3.inference_batch(
                t3_cond=t3_cond_to_use,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            
            # S3Gen: Process items in a single large batch (Tensor Batching)
            # This enables saturating high-end GPUs like B200.
            
            from torch.nn.utils.rnn import pad_sequence

            # 1. Prepare speech tokens (pad to max length)
            valid_tokens_list = []
            valid_token_lens = []
            for i in range(batch_size):
                # drop_invalid_tokens returns 1D tensor [L]
                # We interpret invalid tokens as "EOS" or similar logic
                valid_t = drop_invalid_tokens(speech_tokens[i])
                valid_tokens_list.append(valid_t)
                valid_token_lens.append(len(valid_t))
            
            # Create padded batch [B, T_max]
            # Use 0 as pad value (masked out by flow model)
            s3_input_tokens = pad_sequence(valid_tokens_list, batch_first=True, padding_value=0).to(self.device)
            s3_token_lens = torch.tensor(valid_token_lens, dtype=torch.long, device=self.device)
            
            # 2. Collate reference dicts
            # S3Gen has a static method collate_ref_dicts
            collated_ref_dict = self.s3gen.collate_ref_dicts(item_ref_dicts, device=self.device)
            
            # 3. Batch Inference
            # We use the primary S3Gen instance (no need for copies anymore)
            wav_batch, _ = self.s3gen.inference(
                speech_tokens=s3_input_tokens,
                ref_dict=collated_ref_dict,
                speech_token_lens=s3_token_lens
            )
            # wav_batch is [B, 1, T_wav_max] (padded/ragged output?)
            # S3Gen.inference returns [B, 1, T_max] based on flow inference output.
            # We need to slice each item to its valid length?
            # flow.py inference (my modification) returns batch with max_len but zeros/garbage in padded area?
            # watermarking/post-proc usually ignores padding if we just convert to numpy?
            # But we should probably trim if we want exact results.
            # For now, let's watermark the whole thing (fast) or just slice.
            # Since S3Gen returns [B, 1, T], let's assume valid part is what we want.
            # But actually flow inference returns [B, D, T]. HiFiGAN returns [B, 1, T_upsampled].
            # The upsampled length depends on input len.
            # T_wav = T_mel * 256. 
            # We can calculate valid wav len from token len.
            
            # Post-process results
            results = []
            for i in range(batch_size):
                # Calculate valid wav length
                # From flow.py logic: mel_len = token_len / 50 * 22050 / 256 ... rough approx
                # Or just trust the silence/padding?
                # Let's slice based on s3_token_lens to be clean.
                # Mel len approx = token_len * token_mel_ratio (2?) ?
                # S3Gen logic: token_len2 -> mel_len2 calculation.
                # In flow.py: mel_len2 = (token_len2 / input_frame_rate * 22050 / 256)
                # input_frame_rate=25. 22050/256 ~ 86.
                # So wav_len ~ token_len / 25 * 22050 ~ token_len * 882.
                # This is precise.
                
                # Add a small buffer (e.g., 3 frames/tokens ~ 120ms) to avoid cutting off the release
                # 3 tokens * (1/25) * 22050 = 2646 samples
                BUFFER_TOKENS = 3
                valid_wav_len = int((valid_token_lens[i] + BUFFER_TOKENS) / 25.0 * 22050.0)
                
                # Clamp to actual generated length
                valid_wav_len = min(valid_wav_len, wav_batch.shape[1])
                # Actually HiFiGAN upsamples exactly.
                # Current s3gen returns padded batch.
                
                # Careful: The raw wav tensor might be slightly larger due to padding structure.
                # But cutting at valid_wav_len is safe.
                
                # wav_batch[i] is [T_max] (1D) because wav_batch is [B, T_max]
                curr_wav = wav_batch[i, :valid_wav_len]
                
                wav_numpy = curr_wav.detach().cpu().numpy()
                watermarked_wav = self.watermarker.apply_watermark(wav_numpy, sample_rate=self.sr)
                results.append(torch.from_numpy(watermarked_wav).unsqueeze(0))
            
            # Clear CUDA cache if needed (optional, maybe not for perf)
            # torch.cuda.empty_cache()
            
            return results
