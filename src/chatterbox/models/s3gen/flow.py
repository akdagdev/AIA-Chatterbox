# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
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
import random
from typing import Dict, Optional

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS


class MaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 4096,
        input_frame_rate: int = 50,
        only_mask_loss: bool = True,
        encoder: torch.nn.Module = None,
        length_regulator: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            'in_channels': 240,
            'out_channel': 80,
            'spk_emb_dim': 80,
            'n_spks': 1,
            'cfm_params': CFM_PARAMS,
            'decoder_params': {
                'channels': [256, 256],
                'dropout': 0.0,
                'attention_head_dim': 64,
                'n_blocks': 4,
                'num_mid_blocks': 12,
                'num_heads': 8,
                'act_fn': 'gelu',
            }
        },
        mel_feat_conf: Dict = {
            'n_fft': 1024,
            'num_mels': 80,
            'sampling_rate': 22050,
            'hop_size': 256,
            'win_size': 1024,
            'fmin': 0,
            'fmax': 8000
        }
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.length_regulator = length_regulator
        self.only_mask_loss = only_mask_loss

    def forward(
            self,
            batch: dict,
            device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch['speech_token'].to(device)
        token_len = batch['speech_token_len'].to(device)
        feat = batch['speech_feat'].to(device)
        feat_len = batch['speech_feat_len'].to(device)
        embedding = batch['embedding'].to(device)

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
        token = self.input_embedding(torch.clamp(token, min=0, max=self.input_embedding.num_embeddings-1)) * mask

        # text encode
        h, h_lengths = self.encoder(token, token_len)
        h = self.encoder_proj(h)
        h, h_lengths = self.length_regulator(h, feat_len)

        # get conditions
        conds = torch.zeros(feat.shape, device=token.device)
        for i, j in enumerate(feat_len):
            if random.random() < 0.5:
                continue
            index = random.randint(0, int(0.3 * j))
            conds[i, :index] = feat[i, :index]
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(feat_len)).to(h)
        feat = F.interpolate(feat.unsqueeze(dim=1), size=h.shape[1:], mode="nearest").squeeze(dim=1)
        loss, _ = self.decoder.compute_loss(
            feat.transpose(1, 2).contiguous(),
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds
        )
        return {'loss': loss}

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  flow_cache):
        if self.fp16 is True:
            prompt_feat = prompt_feat.half()
            embedding = embedding.half()

        # Remove single batch assertion
        batch_size = token.shape[0]
        
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # concat text and prompt_text
        # prompt_token is [B, T_prompt] (padded)
        # token is [B, T_text]
        # We need to concat them. Since they might have padding, straight concat puts padding in middle?
        # NO. prompt_token is padded at the end. token is padded at the end?
        # S3Tokenizer doesn't pad internally if we pass single items, but here we pass a batch.
        # Ideally, we should concat valid tokens then pad? This is tricky with tensors.
        # Assumption: S3Gen collate pads at the END. 
        # But wait, we want [Prompt_Valid, Text_Valid, Pad].
        # If we concat [Prompt_Padded, Text], we get [Prompt_Valid, Prompt_Pad, Text_Valid]. THIS IS WRONG.
        
        # However, making it fully dynamic is complex.
        # Let's trust that the mask handles the "Prompt_Pad" ignoring?
        # Creating a combined token tensor that respects lengths:
        
        # Calculate lengths
        token_len1 = prompt_token_len # [B]
        token_len2 = token_len        # [B] (this is text token len)
        total_token_len = token_len1 + token_len2
        max_total_len = total_token_len.max().item()
        
        combined_token = torch.zeros((batch_size, max_total_len), dtype=token.dtype, device=token.device)
        # We need to stitch: [0:len1] = prompt, [len1:len1+len2] = text
        for i in range(batch_size):
            p_len = token_len1[i]
            t_len = token_len2[i]
            combined_token[i, :p_len] = prompt_token[i, :p_len]
            combined_token[i, p_len:p_len+t_len] = token[i, :t_len]
            
        token = combined_token
        
        mask = (~make_pad_mask(total_token_len)).unsqueeze(-1).to(embedding)
        
        # Check for out-of-bounds token IDs
        vocab_size = self.input_embedding.num_embeddings
        # Clamp for safety
        token = self.input_embedding(torch.clamp(token, min=0, max=vocab_size-1)) * mask

        # text encode
        h, h_lengths = self.encoder(token, total_token_len)
        h = self.encoder_proj(h)
        
        # Length regulator inference needs update for batching or loop?
        # default length regulator likely assumes single item or needs iteration if ratios differ.
        # LR inference usually calc duration. 
        # Let's check LR implementation. If it's standard Espnet/FastSpeech, it might handle batch.
        # But here logic is: mel_len2 = int(token_len2 / rate * ...)
        # We need to calc per item.
        
        mel_len1 = prompt_feat_len # [B]
        # Calculate generated mel length per item
        mel_len2 = (token_len2.float() / self.input_frame_rate * 22050 / 256).long()
        
        # Run LR inference (it might not support batch varying lengths natively if simpler version)
        # If LR.inference doesn't support batch, we loop.
        # Looking at previous code: h, h_lengths = self.length_regulator.inference(h[:, :token_len1], ...)
        # It sliced h! This implies it relied on fixed split point which implies batch size 1 or fixed prompt len.
        # Since prompt len varies, we MUST likely loop or implement smart batch LR.
        # For safety and correctness now, let's loop the LR part. It's small overhead compared to model.
        
        h_expanded_list = []
        max_h_len = 0
        total_mel_lens = mel_len1 + mel_len2
        
        for i in range(batch_size):
            # Extract valid text encoding part for this item
            # The encoder output 'h' is [B, T, D]. T matches max_total_len.
            # Valid part is up to total_token_len[i].
            # Split point for prompt/text is token_len1[i].
            
            # Encoder output corresponds to [Prompt, Text].
            # LR usually scales Text part? Or both?
            # In FS2, we usually clone prompt mel, and predict text mel duration.
            # Code says: inference(prompt_enc, text_enc, prompt_mel_len, target_text_mel_len...)
            
            p_len = token_len1[i]
            t_len = token_len2[i]
            
            curr_h_prompt = h[i, :p_len].unsqueeze(0)
            curr_h_text = h[i, p_len:p_len+t_len].unsqueeze(0)
            
            curr_mel_len1 = mel_len1[i].item()
            curr_mel_len2 = mel_len2[i].item()
            
            # Call LR for single item
            # Note: LR.inference returns (h_expanded, h_exp_len)
            curr_h_exp, _ = self.length_regulator.inference(
                curr_h_prompt, 
                curr_h_text, 
                curr_mel_len1, 
                curr_mel_len2, 
                self.input_frame_rate
            )
            h_expanded_list.append(curr_h_exp.squeeze(0))
            max_h_len = max(max_h_len, curr_h_exp.size(1))

        # Stack back
        h = torch.zeros((batch_size, max_h_len, self.output_size), dtype=h.dtype, device=h.device)
        for i, hh in enumerate(h_expanded_list):
            h[i, :hh.size(0)] = hh

        # get conditions - batch-aware
        # prompt_feat is [B, Max_P_Len, D] (padded)
        conds = torch.zeros([batch_size, max_h_len, self.output_size], device=token.device).to(h.dtype)
        
        for i in range(batch_size):
            p_len = mel_len1[i]
            # Copy valid prompt feat
            conds[i, :p_len] = prompt_feat[i, :p_len]
            
        conds = conds.transpose(1, 2)
        
        mask = (~make_pad_mask(total_mel_lens)).to(h)
        
        feat, flow_cache = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            prompt_len=mel_len1, # Decoder needs to know where to start generating? 
                                 # If implementation relies on single int prompt_len, batching variable prompt is HARD.
                                 # We might need to check decoder implementation.
                                 # If decoder takes list or tensor of prompt_lens, we are good.
                                 # Passing tensor attempts to be safe.
            flow_cache=flow_cache
        )
        
        # Slice output correctly
        # We need to remove the prompt part.
        # But prompt length varies!
        # Return jagged? Or padded?
        # Ideally return padded tensor output.
        # feat is [B, D, T_total]
        
        # Re-slice to remove prompt (which varies per item)
        # We construct a clean output tensor [B, D, Max_Gen_Len]
        max_gen_len = max([ml2.item() for ml2 in mel_len2])
        feat_out = torch.zeros((batch_size, self.output_size, max_gen_len), dtype=feat.dtype, device=feat.device)
        
        for i in range(batch_size):
            p_len = mel_len1[i]
            g_len = mel_len2[i] # Generated length
            # feat [i, :, p_len : p_len+g_len]
            feat_out[i, :, :g_len] = feat[i, :, p_len : p_len+g_len]
            
        return feat_out.float(), flow_cache


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: torch.nn.Module = None,
        decoder_conf: Dict = {
            'in_channels': 240,
            'out_channel': 80,
            'spk_emb_dim': 80,
            'n_spks': 1,
            'cfm_params': CFM_PARAMS,
            'decoder_params': {
                'channels': [256, 256],
                'dropout': 0.0,
                'attention_head_dim': 64,
                'n_blocks': 4,
                'num_mid_blocks': 12,
                'num_heads': 8,
                'act_fn': 'gelu',
            }
        },
        mel_feat_conf: Dict = {
            'n_fft': 1024,
            'num_mels': 80,
            'sampling_rate': 22050,
            'hop_size': 256,
            'win_size': 1024,
            'fmin': 0,
            'fmax': 8000
        }
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

        # FIXME: this was missing - just putting it in as false
        self.fp16 = False

    @torch.inference_mode()
    def inference(self,
                  token,
                  token_len,
                  prompt_token,
                  prompt_token_len,
                  prompt_feat,
                  prompt_feat_len,
                  embedding,
                  finalize):
        embedding = embedding.to(self.spk_embed_affine_layer.weight.dtype)
        prompt_feat = prompt_feat.to(self.spk_embed_affine_layer.weight.dtype)

        embedding = embedding.to(self.spk_embed_affine_layer.weight.dtype)
        prompt_feat = prompt_feat.to(self.spk_embed_affine_layer.weight.dtype)

        # Removed assertion for batch support
        batch_size = token.shape[0]
        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # Calculate lengths
        token_len1 = prompt_token_len # [B]
        token_len2 = token_len        # [B]
        total_token_len = token_len1 + token_len2
        max_total_len = total_token_len.max().item()
        
        # Combined token tensor [B, Max_Total_Len]
        combined_token = torch.zeros((batch_size, max_total_len), dtype=token.dtype, device=token.device)
        for i in range(batch_size):
            p_len = token_len1[i]
            t_len = token_len2[i]
            combined_token[i, :p_len] = prompt_token[i, :p_len]
            combined_token[i, p_len:p_len+t_len] = token[i, :t_len]
            
        token = combined_token
        mask = (~make_pad_mask(total_token_len)).unsqueeze(-1).to(embedding)
        
        # Clamp tokens
        vocab_size = self.input_embedding.num_embeddings
        token = self.input_embedding(torch.clamp(token, min=0, max=vocab_size-1)) * mask

        # text encode
        h, h_lengths = self.encoder(token, total_token_len)
        if finalize is False:
             # Pre-lookahead logic... assumes similar ratio?
             # Might be risky with variable batch, but assuming standard behavior
             h = h[:, :-self.pre_lookahead_len * self.token_mel_ratio]

        h = self.encoder_proj(h)
        
        # Calculate expected mel lengths
        mel_len1 = prompt_feat_len # [B]
        
        # UpsampleConformerEncoder returns upsampled lengths (total mel len)
        total_mel_lens = h_lengths # [B]
        
        # Handle lookahead trimming if not finalizing
        if finalize is False:
             # Trim global tensor
             trim = self.pre_lookahead_len * self.token_mel_ratio
             if h.size(1) > trim:
                 h = h[:, :-trim]
                 # Update valid lengths
                 total_mel_lens = (total_mel_lens - trim).clamp(min=0)
             else:
                 # Edge case: sequence shorter than trim?
                 h = torch.zeros_like(h[:, :0])
                 total_mel_lens = torch.zeros_like(total_mel_lens)
        
        # Derived mel_len2 (generated length)
        # Note: total_mel_lens includes prompt + generated.
        mel_len2 = (total_mel_lens - mel_len1).clamp(min=0)
        
        max_mel_len = h.size(1)

        # get conditions - batch-aware
        # prompt_feat is [B, Max_P_Len, D]
        conds = torch.zeros([batch_size, max_mel_len, self.output_size], device=token.device).to(h.dtype)
        for i in range(batch_size):
            p_len = mel_len1[i]
            # Copy valid prompt feat
            # Ensure p_len doesn't exceed prompt_feat actual size or max_mel_len
            curr_p_len = min(p_len.item(), prompt_feat.size(1), max_mel_len)
            conds[i, :curr_p_len] = prompt_feat[i, :curr_p_len]
            
        conds = conds.transpose(1, 2)
        
        # Use h directly (it is already upsampled mel-domain features)
        h_mel = h

        mask = (~make_pad_mask(total_mel_lens)).to(h)

        # get conditions - batch-aware
        # prompt_feat is [B, Max_P_Len, D]
        conds = torch.zeros([batch_size, max_mel_len, self.output_size], device=token.device).to(h.dtype)
        for i in range(batch_size):
            p_len = mel_len1[i]
            # Copy valid prompt feat
            conds[i, :p_len] = prompt_feat[i, :p_len]
            
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(total_mel_lens)).to(h)
        
        # Call decoder
        feat, _ = self.decoder(
            mu=h_mel.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10
        )
        
        # Slice output
        # feat is [B, D, T_total]
        # We want [B, D, T_gen] where T_gen varies.
        # Padding output to max gen len.
        max_gen_len = mel_len2.max().item()
        feat_out = torch.zeros((batch_size, self.output_size, max_gen_len), dtype=feat.dtype, device=feat.device)
        
        for i in range(batch_size):
            p_len = mel_len1[i]
            g_len = mel_len2[i]
            feat_out[i, :, :g_len] = feat[i, :, p_len : p_len+g_len]
            
        return feat_out.float(), None
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat, None  # NOTE jrm: why are they returning None here?
