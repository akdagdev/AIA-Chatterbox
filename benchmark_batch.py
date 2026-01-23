import time
import torch
import torchaudio
from pathlib import Path
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

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
    
    # Test texts of varying lengths
    texts = [
        "Hello, this is a quick test.",
        "Batch processing allows us to synthesize multiple sentences at once, improving GPU utilization significantly.",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we interact with technology.",
        "Short sentence.",
        "Another one.",
        "A slightly longer sentence to add some variety to the batch content.",
        "Testing the limits of the batch size with eight sentences in total."
    ]
    
    batch_size = len(texts)
    print(f"\nBenchmarking with batch size: {batch_size}")
    
    # Warmup
    print("Warming up...")
    model.generate_batch(texts[:2], language_id="en", max_new_tokens=50)
    
    # Run Benchmark
    print("\nStarting generation...")
    if device == "cuda":
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    audio_list = model.generate_batch(
        texts=texts,
        language_id="en",
        max_new_tokens=1000
    )
    
    if device == "cuda":
        torch.cuda.synchronize()
        
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Calculate metrics
    total_audio_duration = 0
    output_dir = Path("benchmark_output")
    output_dir.mkdir(exist_ok=True)
    
    for i, audio in enumerate(audio_list):
        # Audio shape is (1, samples)
        samples = audio.shape[1]
        duration = samples / model.sr
        total_audio_duration += duration
        
        # Save output
        save_audio(audio, output_dir / f"output_{i}.wav", model.sr)
    
    rtf = generation_time / total_audio_duration
    latency = generation_time * 1000  # ms
    throughput = len(texts) / generation_time  # sentences per second
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Batch Size:          {batch_size}")
    print(f"Total Audio Duration: {total_audio_duration:.2f}s")
    print(f"Generation Time:      {generation_time:.2f}s")
    print(f"RTF (Time/Audio):     {rtf:.4f} " + ("(Lower is better)" if rtf < 1.0 else "(Higher than real-time)"))
    print(f"Inverse RTF:          {1/rtf:.2f}x real-time")
    print(f"Latency:              {latency:.2f}ms")
    print(f"Throughput:           {throughput:.2f} sentences/sec")
    print("="*40)
    print(f"\nAudio files saved to {output_dir.absolute()}")

if __name__ == "__main__":
    benchmark()
