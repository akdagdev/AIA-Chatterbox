import torch
from typing import Optional, Tuple, Callable, Any, Dict

from transformers.cache_utils import StaticLayer, StaticCache, Cache

TOKEN_LIMIT = 1500


class FlexibleStaticLayer(StaticLayer):
    """StaticLayer that pre-allocates keys/values at max_batch_size.

    The default StaticLayer lazily determines batch size from the first key_states
    passed to update(). If a small-batch call (e.g. effective=2) initialises the
    layer first, a later larger-batch call (effective=4) will fail in index_copy_()
    because all non-indexed dims must match between source and destination:

        Destination slice [2, 16, 64]  ≠  Source slice [4, 16, 64]

    This subclass overrides lazy_initialization to allocate at _max_batch regardless
    of the actual key_states batch size, and overrides update to write/read only the
    [:actual_batch] slice of the buffer, making any batch ≤ _max_batch safe.
    """

    def __init__(self, max_cache_len: int, max_batch_size: int):
        super().__init__(max_cache_len)
        self._max_batch = max_batch_size

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        _, self.num_heads, _, self.head_dim = key_states.shape
        self.max_batch_size = self._max_batch
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.zeros(
            (self._max_batch, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype, device=self.device,
        )
        self.values = torch.zeros(
            (self._max_batch, self.num_heads, self.max_cache_len, self.head_dim),
            dtype=self.dtype, device=self.device,
        )
        try:
            from torch._dynamo import is_compiling as _is_compiling
            if not _is_compiling():
                torch._dynamo.mark_static_address(self.keys)
                torch._dynamo.mark_static_address(self.values)
        except Exception:
            pass
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs=None):
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(key_states.shape[-2], device=self.device)

        # Slice to actual_batch so index_copy_'s dim-0 requirement is satisfied
        actual_batch = key_states.shape[0]
        try:
            self.keys[:actual_batch].index_copy_(2, cache_position, key_states)
            self.values[:actual_batch].index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # MPS fallback
            self.keys[:actual_batch, :, cache_position] = key_states
            self.values[:actual_batch, :, cache_position] = value_states

        return self.keys[:actual_batch], self.values[:actual_batch]


class FlexibleStaticCache(StaticCache):
    """StaticCache composed of FlexibleStaticLayer instances.

    Allows a single shared cache to serve any effective batch size ≤ max_batch_size
    without OOM from per-size pre-allocation.
    """

    def __init__(self, config, max_batch_size: int, max_cache_len: int, **kwargs):
        # Build layers manually, bypassing StaticCache.__init__ which creates
        # plain StaticLayer objects that don't handle variable batch.
        config_text = config.get_text_config(decoder=True)
        num_layers = config_text.num_hidden_layers
        layers = [
            FlexibleStaticLayer(max_cache_len=max_cache_len, max_batch_size=max_batch_size)
            for _ in range(num_layers)
        ]
        # Initialise via Cache.__init__ (grandparent) — passes layers through
        Cache.__init__(self, layers=layers)

# Set externally by the host application (e.g. AI-Server) to prevent CUDA graph
# captures from running concurrently with other CUDA operations (like STT inference).
# When set, the lock is acquired ONLY during graph capture (~10-50ms per bucket),
# NOT during graph replay. This allows normal TTS generation and STT to run freely
# in parallel, blocking STT only during the brief first-time capture of each bucket.
CUDA_CAPTURE_LOCK = None


def get_next_bucket(
    seq_len: int, bucket_size: int = 250, max_bucket: int = TOKEN_LIMIT
) -> int:
    bucket = ((seq_len - 1) // bucket_size + 1) * bucket_size
    return min(bucket, max_bucket)


class T3StepCUDAGraphWrapper:
    """
    A wrapper class that automatically captures and replays CUDA graphs for optimized inference
    with support for bucketing to handle dynamic max_position values.

    Maintains separate graphs for different bucket sizes while sharing kv_cache and memory.
    """

    def __init__(
        self,
        generate_token: Callable,
        patched_model: Any,
        kv_cache: Any,
        repetition_penalty_processor: Any,
        min_p_warper: Any,
        top_p_warper: Any,
        alignment_stream_analyzer: Any = None,
    ):
        """
        Initialize the CUDA graph wrapper with bucketing support.

        Args:
            generate_token: The function to wrap with CUDA graph
            patched_model: The model instance
            kv_cache: The key-value cache (shared across all buckets)
            repetition_penalty_processor: Repetition penalty processor
            min_p_warper: Min-p warper
            top_p_warper: Top-p warper
        """
        self.generate_token = generate_token
        self.patched_model = patched_model
        self.kv_cache = kv_cache  # Shared across all buckets
        self.repetition_penalty_processor = repetition_penalty_processor
        self.min_p_warper = min_p_warper
        self.top_p_warper = top_p_warper

        # Dictionary to store graphs and static tensors for each bucket
        self._bucket_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._bucket_static_tensors: Dict[int, dict] = {}

        # Track which buckets have been captured
        self._captured_buckets = set()

        self.dtype = patched_model.dtype
        self.alignment_stream_analyzer = alignment_stream_analyzer

    def guard(self):
        """
        Validate critical parameters have not changed, such as effective batch size or dtype.
        """
        assert self.patched_model.dtype == self.dtype
        pass

    def _capture_graph_for_bucket(
        self,
        bucket_key: int,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        stride_length: int,
        max_position: Optional[int] = None,
    ) -> None:
        """Capture the CUDA graph for a specific bucket."""
        print(
            f"Capturing CUDA graph for bucket {bucket_key} (max_position: {max_position})"
        )

        lock = CUDA_CAPTURE_LOCK
        if lock:
            lock.acquire()
        try:
            # Create new graph for this bucket
            self._bucket_graphs[bucket_key] = torch.cuda.CUDAGraph()

            # Initialize static tensors dictionary for this bucket
            static_tensors = {}

            # Clone static tensors for this bucket
            static_tensors["speech_embedding_cache"] = speech_embedding_cache.clone()
            static_tensors["output_logits"] = output_logits.clone()
            static_tensors["i_tensor"] = i_tensor.clone()
            static_tensors["batch_idx"] = batch_idx.clone()
            static_tensors["speech_pos_embedding_cache"] = (
                speech_pos_embedding_cache.clone()
            )
            static_tensors["generated_ids"] = generated_ids
            static_tensors["cfg_weight"] = cfg_weight
            static_tensors["temperature"] = temperature
            static_tensors["stride_length"] = stride_length
            static_tensors["max_position"] = bucket_key

            # Capture the graph
            with torch.inference_mode():
                with torch.cuda.graph(self._bucket_graphs[bucket_key]):
                    static_tensors["out_1"], static_tensors["out_2"] = self.generate_token(
                        static_tensors["speech_embedding_cache"],
                        static_tensors["output_logits"],
                        static_tensors["i_tensor"],
                        static_tensors["batch_idx"],
                        static_tensors["speech_pos_embedding_cache"],
                        static_tensors["generated_ids"],
                        static_tensors["cfg_weight"],
                        static_tensors["temperature"],
                        self.repetition_penalty_processor,
                        self.min_p_warper,
                        self.top_p_warper,
                        self.patched_model,
                        self.kv_cache,  # Shared kv_cache
                        static_tensors["stride_length"],
                        static_tensors["max_position"],
                        self.alignment_stream_analyzer,
                    )

            # Store static tensors for this bucket
            self._bucket_static_tensors[bucket_key] = static_tensors
        finally:
            if lock:
                lock.release()
        self._captured_buckets.add(bucket_key)

    def __call__(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        repetition_penalty_processor: Any = None,
        min_p_warper: Any = None,
        top_p_warper: Any = None,
        patched_model: Any = None,
        kv_cache: Any = None,
        stride_length: int = 1,
        max_position: Optional[int] = None,
        alignment_stream_analyzer: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine which bucket to use
        bucket_key = max_position or TOKEN_LIMIT

        # Check if we need to capture a graph for this bucket
        if bucket_key not in self._captured_buckets:
            self._capture_graph_for_bucket(
                bucket_key,
                speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                speech_pos_embedding_cache,
                generated_ids,
                cfg_weight,
                temperature,
                stride_length,
                max_position,
            )
        else:
            # Update static tensors for this bucket and replay
            static_tensors = self._bucket_static_tensors[bucket_key]

            static_tensors["speech_embedding_cache"].copy_(speech_embedding_cache)
            static_tensors["output_logits"].copy_(output_logits)
            static_tensors["i_tensor"].copy_(i_tensor)
            static_tensors["batch_idx"].copy_(batch_idx)
            static_tensors["speech_pos_embedding_cache"].copy_(
                speech_pos_embedding_cache
            )
            static_tensors["generated_ids"].copy_(generated_ids)
            static_tensors["cfg_weight"] = cfg_weight
            static_tensors["temperature"] = temperature
            static_tensors["stride_length"] = stride_length
            static_tensors["max_position"] = max_position

            # Replay the graph for this bucket
            self._bucket_graphs[bucket_key].replay()

        # Return outputs from the appropriate bucket
        static_tensors = self._bucket_static_tensors[bucket_key]
        return (
            static_tensors["out_1"],
            static_tensors["out_2"],
            static_tensors["generated_ids"],
        )

    def reset(self, bucket_key: Optional[int] = None) -> None:
        if bucket_key is not None:
            # Reset specific bucket
            if bucket_key in self._bucket_graphs:
                del self._bucket_graphs[bucket_key]
            if bucket_key in self._bucket_static_tensors:
                del self._bucket_static_tensors[bucket_key]
            self._captured_buckets.discard(bucket_key)
            print(f"Reset bucket {bucket_key}")
        else:
            # Reset all buckets
            self._bucket_graphs.clear()
            self._bucket_static_tensors.clear()
            self._captured_buckets.clear()
            print("Reset all buckets")

    def mark_new_generation(self):
        self.kv_cache.reset()


# Bucket sizes for prefill seq_len. Must cover all realistic input lengths.
# Typical T3 single-mode seq_len = conditioning (~30) + text (~20-200) + BOS (1) ≈ 50-230.
PREFILL_BUCKETS = [64, 128, 256, 512]


class PrefillCUDAGraphWrapper:
    """
    CUDA graph for the T3 initial forward pass (prefill).

    The prefill runs once per generation to process text+conditioning through all Llama
    layers and fill the KV cache. seq_len varies per call (different text lengths), so
    we bucket to fixed sizes to make capture possible.

    Strategy: right-pad inputs_embeds with zeros to the bucket size.
    - Real tokens at positions 0..seq_len-1 get correct RoPE.
    - Padded zeros produce zero K/V (Llama uses bias=False throughout).
    - Causal mask prevents real tokens from attending to future padded positions.
    - Output is captured for all bucket positions; post-graph we slice [:, seq_len-1, :].

    The kv_cache is the shared _direct_cache from T3. After the prefill graph runs,
    KV slots 0..bucket-1 are filled (real at 0..seq_len-1, zeros at seq_len..bucket-1).
    The generation loop starts at cache_position=seq_len and overwrites any zero slots.
    """

    def __init__(self, patched_model, kv_cache):
        self.patched_model = patched_model
        self.kv_cache = kv_cache
        self._graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._static: Dict[int, dict] = {}
        self._captured: set = set()

    @staticmethod
    def _get_bucket(seq_len: int) -> Optional[int]:
        for b in PREFILL_BUCKETS:
            if seq_len <= b:
                return b
        return None  # seq_len too long for any bucket → eager fallback

    def __call__(
        self,
        inputs_embeds: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        inputs_embeds: [batch, ≥seq_len, hidden] — may be pre-padded to TOKEN_LIMIT
        seq_len: actual sequence length (without TOKEN_LIMIT padding)
        Returns: logits [batch, 1, vocab_size] for position seq_len-1
        """
        bucket = self._get_bucket(seq_len)
        if bucket is None:
            # Fallback for unusually long prompts (seq_len > 512): run eager
            cache_position = torch.arange(seq_len, device=inputs_embeds.device)
            out = self.patched_model(
                inputs_embeds=inputs_embeds[:, :seq_len, :],
                past_key_values=self.kv_cache,
                cache_position=cache_position,
            )
            return out[:, -1:, :]

        if bucket not in self._captured:
            self._capture(bucket, inputs_embeds, seq_len)
        else:
            static = self._static[bucket]
            # Zero the padding region, copy real embeddings
            static["embeds"].zero_()
            static["embeds"][:, :seq_len, :].copy_(inputs_embeds[:, :seq_len, :])
            self._graphs[bucket].replay()

        # Post-graph slice: extract logit for the last real token (Python-side, free)
        return self._static[bucket]["out"][:, seq_len - 1 : seq_len, :].clone()

    def _capture(self, bucket: int, inputs_embeds: torch.Tensor, seq_len: int):
        batch, hidden = inputs_embeds.shape[0], inputs_embeds.shape[2]
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        static_embeds = torch.zeros(batch, bucket, hidden, dtype=dtype, device=device)
        static_embeds[:, :seq_len, :].copy_(inputs_embeds[:, :seq_len, :])
        static_cache_pos = torch.arange(bucket, device=device, dtype=torch.long)

        lock = CUDA_CAPTURE_LOCK
        if lock:
            lock.acquire()
        try:
            with torch.inference_mode():
                # 3 warmup passes to heat cuBLAS kernels before capture
                for _ in range(3):
                    self.kv_cache.reset()
                    self.patched_model(
                        inputs_embeds=static_embeds,
                        past_key_values=self.kv_cache,
                        cache_position=static_cache_pos,
                    )
                self.kv_cache.reset()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = self.patched_model(
                        inputs_embeds=static_embeds,
                        past_key_values=self.kv_cache,
                        cache_position=static_cache_pos,
                    )
            torch.cuda.synchronize()
        finally:
            if lock:
                lock.release()

        self._graphs[bucket] = graph
        self._static[bucket] = {
            "embeds": static_embeds,
            "cache_pos": static_cache_pos,
            "out": static_out,
        }
        self._captured.add(bucket)

    def reset(self):
        self._graphs.clear()
        self._static.clear()
        self._captured.clear()


class T3BatchStepCUDAGraphWrapper:
    """
    CUDA Graph wrapper for batch token generation.
    Similar to T3StepCUDAGraphWrapper but handles batch_size > 1 and finished_mask.
    """

    def __init__(
        self,
        generate_token_batch: Callable,
        patched_model: Any,
        kv_cache: Any,
        repetition_penalty_processor: Any,
        min_p_warper: Any,
        top_p_warper: Any,
        input_batch_size: int,
        stop_token_id: int,
        stop_token_tensor: torch.Tensor,  # Pre-allocated tensor
    ):
        self.generate_token_batch = generate_token_batch
        self.patched_model = patched_model
        self.kv_cache = kv_cache
        self.repetition_penalty_processor = repetition_penalty_processor
        self.min_p_warper = min_p_warper
        self.top_p_warper = top_p_warper
        self.input_batch_size = input_batch_size
        self.stop_token_id = stop_token_id
        self.stop_token_tensor = stop_token_tensor

        self._bucket_graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._bucket_static_tensors: Dict[int, dict] = {}
        self._captured_buckets = set()
        self.dtype = patched_model.dtype

    def guard(self):
        assert self.patched_model.dtype == self.dtype

    def _capture_graph_for_bucket(
        self,
        bucket_key: int,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        finished_mask: torch.Tensor,
        max_position: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> None:
        print(f"Capturing CUDA graph for batch bucket {bucket_key}")

        lock = CUDA_CAPTURE_LOCK
        if lock:
            lock.acquire()
        try:
            self._bucket_graphs[bucket_key] = torch.cuda.CUDAGraph()
            static_tensors = {}

            static_tensors["speech_embedding_cache"] = speech_embedding_cache.clone()
            static_tensors["output_logits"] = output_logits.clone()
            static_tensors["i_tensor"] = i_tensor.clone()
            static_tensors["batch_idx"] = batch_idx.clone()
            static_tensors["speech_pos_embedding_cache"] = speech_pos_embedding_cache.clone()
            static_tensors["generated_ids"] = generated_ids.clone()
            static_tensors["cfg_weight"] = cfg_weight
            static_tensors["temperature"] = temperature
            static_tensors["finished_mask"] = finished_mask.clone()
            static_tensors["max_position"] = bucket_key
            static_tensors["attention_mask"] = attention_mask.clone() if attention_mask is not None else None
            static_tensors["cache_position"] = cache_position.clone() if cache_position is not None else None

            # Warmup: run 3 eager passes before graph capture to allow cuBLAS to
            # auto-select kernels and allocate workspace for this batch size.
            # Without warmup, cuBLAS tries to allocate inside torch.cuda.graph()
            # which is forbidden, corrupting the capture-time output (all EOS).
            # Save/restore the KV cache slot written during warmup so the capture
            # sees the correct pre-warmup state.
            if cache_position is not None:
                warmup_cache_pos = int(cache_position.item())
                kv_slot_saves = [
                    (layer.keys[:, :, warmup_cache_pos, :].clone(),
                     layer.values[:, :, warmup_cache_pos, :].clone())
                    for layer in self.kv_cache.layers
                    if layer.is_initialized
                ]
            else:
                kv_slot_saves = None

            with torch.inference_mode():
                warmup_gen_ids = static_tensors["generated_ids"].clone()
                for wi in range(3):
                    warmup_out1, _ = self.generate_token_batch(
                        static_tensors["speech_embedding_cache"],
                        static_tensors["output_logits"].clone(),
                        static_tensors["i_tensor"],
                        static_tensors["batch_idx"],
                        static_tensors["speech_pos_embedding_cache"],
                        warmup_gen_ids,
                        static_tensors["cfg_weight"],
                        static_tensors["temperature"],
                        self.repetition_penalty_processor,
                        self.min_p_warper,
                        self.top_p_warper,
                        self.patched_model,
                        self.kv_cache,
                        self.input_batch_size,
                        static_tensors["finished_mask"],
                        self.stop_token_id,
                        self.stop_token_tensor,
                        static_tensors["max_position"],
                        static_tensors["attention_mask"],
                        static_tensors["cache_position"],
                    )
                    warmup_gen_ids.copy_(static_tensors["generated_ids"])
            del warmup_gen_ids

            # Restore the KV slot overwritten during warmup
            if kv_slot_saves is not None:
                initialized_layers = [l for l in self.kv_cache.layers if l.is_initialized]
                for (key_s, val_s), layer in zip(kv_slot_saves, initialized_layers):
                    layer.keys[:, :, warmup_cache_pos, :].copy_(key_s)
                    layer.values[:, :, warmup_cache_pos, :].copy_(val_s)
                del kv_slot_saves

            with torch.inference_mode():
                with torch.cuda.graph(self._bucket_graphs[bucket_key]):
                    static_tensors["out_1"], static_tensors["out_2"] = self.generate_token_batch(
                        static_tensors["speech_embedding_cache"],
                        static_tensors["output_logits"],
                        static_tensors["i_tensor"],
                        static_tensors["batch_idx"],
                        static_tensors["speech_pos_embedding_cache"],
                        static_tensors["generated_ids"],
                        static_tensors["cfg_weight"],
                        static_tensors["temperature"],
                        self.repetition_penalty_processor,
                        self.min_p_warper,
                        self.top_p_warper,
                        self.patched_model,
                        self.kv_cache,
                        self.input_batch_size,
                        static_tensors["finished_mask"],
                        self.stop_token_id,
                        self.stop_token_tensor,
                        static_tensors["max_position"],
                        static_tensors["attention_mask"],
                        static_tensors["cache_position"],
                    )

            self._bucket_static_tensors[bucket_key] = static_tensors
            self._captured_buckets.add(bucket_key)
            torch.cuda.synchronize()
        finally:
            if lock:
                lock.release()

    def __call__(
        self,
        speech_embedding_cache: torch.Tensor,
        output_logits: torch.Tensor,
        i_tensor: torch.Tensor,
        batch_idx: torch.Tensor,
        speech_pos_embedding_cache: torch.Tensor,
        generated_ids: torch.Tensor,
        cfg_weight: float,
        temperature: float,
        repetition_penalty_processor: Any = None,
        min_p_warper: Any = None,
        top_p_warper: Any = None,
        patched_model: Any = None,
        kv_cache: Any = None,
        input_batch_size: int = 1,
        finished_mask: torch.Tensor = None,
        stop_token_id: int = None,
        stop_token_tensor: torch.Tensor = None,
        max_position: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bucket_key = max_position or TOKEN_LIMIT

        if bucket_key not in self._captured_buckets:
            # Save generated_ids before capture — CUDA graph context makes multinomial
            # return token 0 during capture, corrupting static_tensors["out_1"].
            original_gen_ids = generated_ids.clone()

            self._capture_graph_for_bucket(
                bucket_key,
                speech_embedding_cache,
                output_logits,
                i_tensor,
                batch_idx,
                speech_pos_embedding_cache,
                generated_ids,
                cfg_weight,
                temperature,
                finished_mask,
                max_position,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )

            # Graph is captured for future replays. Run one eager pass with the
            # original (pre-capture) generated_ids to get the real first token.
            with torch.inference_mode():
                real_out_1, real_out_2 = self.generate_token_batch(
                    speech_embedding_cache,
                    output_logits,
                    i_tensor,
                    batch_idx,
                    speech_pos_embedding_cache,
                    original_gen_ids,
                    cfg_weight,
                    temperature,
                    self.repetition_penalty_processor,
                    self.min_p_warper,
                    self.top_p_warper,
                    self.patched_model,
                    self.kv_cache,
                    self.input_batch_size,
                    finished_mask,
                    self.stop_token_id,
                    self.stop_token_tensor,
                    max_position,
                    attention_mask,
                    cache_position,
                )
            # Sync generated_ids with the real token so the loop state is correct
            generated_ids.copy_(original_gen_ids)
            return (real_out_1, real_out_2)
        else:
            static_tensors = self._bucket_static_tensors[bucket_key]
            # Only copy tensors that change between iterations.
            # speech_embedding_cache, speech_pos_embedding_cache, batch_idx, cfg_weight,
            # temperature are constant across all steps — captured once, never re-copied.
            static_tensors["output_logits"].copy_(output_logits)
            static_tensors["i_tensor"].copy_(i_tensor)
            static_tensors["generated_ids"].copy_(generated_ids)
            static_tensors["finished_mask"].copy_(finished_mask)
            if static_tensors["attention_mask"] is not None and attention_mask is not None:
                static_tensors["attention_mask"].copy_(attention_mask)
            if static_tensors["cache_position"] is not None and cache_position is not None:
                static_tensors["cache_position"].copy_(cache_position)

            self._bucket_graphs[bucket_key].replay()

            # Copy static generated_ids back to original after replay
            generated_ids.copy_(static_tensors["generated_ids"])

        static_tensors = self._bucket_static_tensors[bucket_key]
        return (
            static_tensors["out_1"],
            static_tensors["out_2"],
        )

    def reset(self, bucket_key: Optional[int] = None) -> None:
        if bucket_key is not None:
            if bucket_key in self._bucket_graphs:
                del self._bucket_graphs[bucket_key]
            if bucket_key in self._bucket_static_tensors:
                del self._bucket_static_tensors[bucket_key]
            self._captured_buckets.discard(bucket_key)
        else:
            self._bucket_graphs.clear()
            self._bucket_static_tensors.clear()
            self._captured_buckets.clear()

    def mark_new_generation(self):
        self.kv_cache.reset()
