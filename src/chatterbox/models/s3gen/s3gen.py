# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional

from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    """
    CosyVoice2's CFM decoder maps S3 speech tokens to mel-spectrograms.

    TODO: make these modules configurable?
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram # TODO: make it a torch module?
        self.speaker_encoder = CAMPPlus()  # use default args

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
        )
        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device
    
    @property
    def dtype(self):
        params = self.tokenizer.parameters()
        return next(params).dtype

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: cosydec received ref longer than 10s")

        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)

        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(device)
        ref_mels_24_len = torch.tensor([ref_mels_24.size(1)], dtype=torch.long, device=device)

        # Resample to 16kHz
        ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav).to(device)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        # Tokenize 16khz reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16)

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )

    @staticmethod
    def collate_ref_dicts(ref_dicts: list, device="cpu"):
        """
        Collate a list of ref_dicts into a single batched ref_dict.
        Handles padding for prompt_token and prompt_feat.
        """
        if not ref_dicts:
            return None
        
        batch_size = len(ref_dicts)
        
        # 1. Stack embeddings (fixed size vector)
        embedding = torch.cat([d['embedding'] for d in ref_dicts], dim=0).to(device)
        
        # 2. Pad and stack prompt_token
        # prompt_token is [1, T]
        prompt_tokens = [d['prompt_token'] for d in ref_dicts]
        max_token_len = max([t.size(1) for t in prompt_tokens])
        padded_tokens = []
        prompt_token_lens = []
        
        for t in prompt_tokens:
            curr_len = t.size(1)
            prompt_token_lens.append(curr_len)
            if curr_len < max_token_len:
                # Pad with 0 (or specific pad token if needed, 0 is safe for S3)
                t = F.pad(t, (0, max_token_len - curr_len), value=0)
            padded_tokens.append(t)
            
        prompt_token = torch.cat(padded_tokens, dim=0).to(device)
        prompt_token_len = torch.tensor(prompt_token_lens, dtype=torch.long, device=device)
        
        # 3. Pad and stack prompt_feat
        # prompt_feat is [1, T, 80] or similar (B, T, D) -> actually (B, T, D) based on usage
        # Let's check shape: ref_mels_24 is (1, T, 80)
        prompt_feats = [d['prompt_feat'] for d in ref_dicts]
        max_feat_len = max([f.size(1) for f in prompt_feats])
        padded_feats = []
        prompt_feat_lens = []
        
        for f in prompt_feats:
            curr_len = f.size(1)
            prompt_feat_lens.append(curr_len)
            if curr_len < max_feat_len:
                # Pad last dim (time) with 0
                # F.pad signature for 3D (B, T, D): (pad_left, pad_right, pad_top, pad_bottom, ...)
                # for (B, T, D), we want to pad dim 1 (T).
                # Pytorch padding starts from last dim.
                # last dim is D (no pad), 2nd last is T.
                # pad format: (d_left, d_right, t_left, t_right)
                f = F.pad(f, (0, 0, 0, max_feat_len - curr_len), value=0)
            padded_feats.append(f)
            
        prompt_feat = torch.cat(padded_feats, dim=0).to(device)
        prompt_feat_len = torch.tensor(prompt_feat_lens, dtype=torch.long, device=device)
        
        return dict(
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
        )

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.

        NOTE:
        - The speaker encoder accepts 16 kHz waveform.
        - S3TokenizerV2 accepts 16 kHz waveform.
        - The mel-spectrogram for the reference assumes 24 kHz input signal.
        - This function is designed for batch_size=1 only.

        Args
        ----
        - `speech_tokens`: S3 speech tokens [B=1, T]
        - `ref_wav`: reference waveform (`torch.Tensor` with shape=[B=1, T])
        - `ref_sr`: reference sample rate
        - `finalize`: whether streaming is finished or not. Note that if False, the last 3 tokens will be ignored.
        - `speech_token_lens`: optional tensor of lengths for `speech_tokens`. If not provided, full length is used.
        """
        assert (ref_wav is None) ^ (ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # type/device/dtype casting (all values will be numpy if it's from a prod API call)
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    # Cast to model device and dtype (supports FP16/FP32)
                    if ref_dict[rk].is_floating_point():
                        ref_dict[rk] = ref_dict[rk].to(device=self.device, dtype=self.dtype)
                    else:
                        ref_dict[rk] = ref_dict[rk].to(self.device)

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        # Batch size support
        batch_size = speech_tokens.size(0)
        
        # Calculate lengths
        if speech_token_lens is None:
            speech_token_lens = torch.full((batch_size,), speech_tokens.size(1), dtype=torch.long, device=self.device)
        else:
            speech_token_lens = speech_token_lens.to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            **ref_dict,
        )
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    The decoder of CosyVoice2 is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.

    TODO: make these modules configurable?
    """

    def __init__(self, compile_model: bool = False):
        super().__init__()

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False) # (buffers get automatic device casting)

        # Optional torch.compile for performance optimization
        if compile_model:
            self.flow = torch.compile(self.flow, mode="reduce-overhead")
            self.mel2wav = torch.compile(self.mel2wav, mode="reduce-overhead")

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False
    ):
        output_mels = super().forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        if not self.training:
            # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            fade_len = min(output_wavs.size(1), len(self.trim_fade))
            output_wavs[:, :fade_len] *= self.trim_fade[:fade_len]

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
    ):
        output_mels = super().forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize, speech_token_lens=speech_token_lens)
        return output_mels.to(self.dtype)

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(self.device)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        cache_source: torch.Tensor = None, # NOTE: this arg is for streaming, it can probably be removed here
        finalize: bool = True,
        no_trim: bool = False,
        speech_token_lens: Optional[torch.Tensor] = None,
    ):
        output_mels = self.flow_inference(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize, speech_token_lens=speech_token_lens)
        output_wavs, output_sources = self.hift_inference(output_mels, cache_source)

        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        if not no_trim:
            fade_len = min(output_wavs.size(1), len(self.trim_fade))
            output_wavs[:, :fade_len] *= self.trim_fade[:fade_len]

        return output_wavs, output_sources
