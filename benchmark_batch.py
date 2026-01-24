import time
import torch
import torchaudio
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Timing tracking
timings = {}

def save_audio(tensor, path, sr=24000):
    """Save audio tensor to file."""
    # Ensure tensor is (channels, time)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    
    # torchaudio expects (channels, time)
    # Check if tensor is on GPU
    tensor = tensor.detach().cpu()
    
    torchaudio.save(path, tensor, sr)
    print(f"Saved audio to {path}")

def benchmark():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    
    # Import SpeechRequest
    from chatterbox.mtl_tts import SpeechRequest
    
    # Test texts with mixed languages and prompts
    # Note: ensure benchmark_output/output_0.wav etc exist or use None
    requests = [
        SpeechRequest(text="Hello, how are you?", language_id="en", audio_prompt_path=None),
        SpeechRequest(text="Merhaba, nasılsın?", language_id="tr", audio_prompt_path=None),
        SpeechRequest(text="Hola, ¿cómo estás?", language_id="es", audio_prompt_path=None),
        SpeechRequest(text="Bonjour, comment allez-vous?", language_id="fr", audio_prompt_path=None),
    ]

    requests = requests * 4
    
    batch_size = len(requests)
    print(f"\nBenchmarking with batch size: {batch_size}")
    
    # Warmup: Run with FULL batch size to ensure CUDA graphs are captured correctly
    print("Warming up with full batch size...")
    # Generate batch accepts List[SpeechRequest] as 'texts' argument
    model.generate_batch(texts=requests, language_id=None, max_new_tokens=400, cfg_weight=0.3)
    print("Warmup complete - CUDA graphs captured")
    
    # Run Benchmark with detailed timing
    print("\nStarting generation with profiling (Run 1 - cache warming)...")
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Inject timing into the model
    original_inference_batch = model.t3.inference_batch
    original_s3gen_inference = model.s3gen.inference
    
    t3_times = []
    s3gen_times = []
    
    def timed_t3_inference_batch(*args, **kwargs):
        start = time.time()
        if device == "cuda":
            torch.cuda.synchronize()
        result = original_inference_batch(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        t3_times.append(time.time() - start)
        return result
    
    def timed_s3gen_inference(*args, **kwargs):
        start = time.time()
        if device == "cuda":
            torch.cuda.synchronize()
        result = original_s3gen_inference(*args, **kwargs)
        if device == "cuda":
            torch.cuda.synchronize()
        s3gen_times.append(time.time() - start)
        return result
    
    model.t3.inference_batch = timed_t3_inference_batch
    model.s3gen.inference = timed_s3gen_inference
    
    # Run 1 (for any remaining cache warming)
    model.generate_batch(
        texts=requests,
        language_id=None,
        max_new_tokens=400,
        cfg_weight=0.3
    )
    
    # Reset timing for Run 2
    t3_times.clear()
    s3gen_times.clear()
    if device == "cuda":
        torch.cuda.synchronize()
    
    print("Starting generation with profiling (Run 2 - actual measurement)...")
    start_time = time.time()
    
    audio_list = model.generate_batch(
        texts=requests,
        language_id=None,
        max_new_tokens=400,
        cfg_weight=0.3
    )
    
    if device == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Restore original methods
    model.t3.inference_batch = original_inference_batch
    model.s3gen.inference = original_s3gen_inference
    
    # Calculate metrics
    total_audio_duration = 0
    output_dir = Path("benchmark_output")
    output_dir.mkdir(exist_ok=True)
    
    for i, audio in enumerate(audio_list):
        samples = audio.shape[1]
        duration = samples / model.sr
        total_audio_duration += duration
        save_audio(audio, output_dir / f"output_{i}.wav", model.sr)
    
    rtf = generation_time / total_audio_duration
    latency = generation_time * 1000
    throughput = len(requests) / generation_time
    
    # Timing breakdown
    t3_total = sum(t3_times)
    s3gen_total = sum(s3gen_times)
    other_time = generation_time - t3_total - s3gen_total
    
    print("\n" + "="*50)
    print("TIMING BREAKDOWN")
    print("="*50)
    print(f"T3 (Text-to-Token):    {t3_total:.2f}s ({t3_total/generation_time*100:.1f}%)")
    print(f"  - Called {len(t3_times)} time(s)")
    print(f"S3Gen (Token-to-Wav):  {s3gen_total:.2f}s ({s3gen_total/generation_time*100:.1f}%)")
    print(f"  - Called {len(s3gen_times)} time(s)")
    print(f"  - Avg per call: {s3gen_total/len(s3gen_times) if s3gen_times else 0:.3f}s")
    print(f"Other (tokenization, watermarking, etc.): {other_time:.2f}s ({other_time/generation_time*100:.1f}%)")
    print("="*50)
    
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    print(f"Batch Size:          {batch_size}")
    print(f"Total Audio Duration: {total_audio_duration:.2f}s")
    print(f"Generation Time:      {generation_time:.2f}s")
    print(f"RTF (Time/Audio):     {rtf:.4f} " + ("(Lower is better)" if rtf < 1.0 else "(Higher than real-time)"))
    print(f"Inverse RTF:          {1/rtf:.2f}x real-time")
    print(f"Latency:              {latency:.2f}ms")
    print(f"Throughput:           {throughput:.2f} sentences/sec")
    print("="*50)
    print(f"\nAudio files saved to {output_dir.absolute()}")

if __name__ == "__main__":
    benchmark()
