import time
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

def text_generator(total_sentences=50):
    sentences = [
        "This is the first sentence.",
        "Here is the second one, coming right up.",
        "Streaming allows for lower latency.",
        "The server can process requests continuously.",
        "We are testing the throughput and latency now."
    ]
    for i in range(total_sentences):
        yield sentences[i % len(sentences)]

def benchmark_stream():
    print("Loading model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxMultilingualTTS.from_pretrained(device)
    
    # Warmup
    print("Warming up...")
    # Just consume the generator
    for _ in model.generate_stream(["Warmup"], language_id="en", micro_batch_size=1):
        pass
    
    if device == "cuda":
        torch.cuda.synchronize()

    print("\nStarting Streaming Benchmark...")
    start_time = time.time()
    first_audio_time = None
    
    # Run stream with micro-batch size 8 (high throughput)
    # prefetch_batches=1 means getting results from previous batch while current runs
    stream = model.generate_stream(
        text_generator(100), 
        language_id="en", 
        micro_batch_size=8,
        prefetch_batches=1,
        cfg_weight=0 # Disable CFG for speed in benchmark
    )
    
    total_audio_duration = 0.0
    
    count = 0
    for i, audio in enumerate(stream):
        # Calculate duration of this chunk
        samples = audio.shape[1]
        duration = samples / model.sr
        total_audio_duration += duration
        
        if i == 0:
            if device == "cuda":
                torch.cuda.synchronize() # Ensure it's truly done for timing
            first_audio_time = time.time()
            latency = (first_audio_time - start_time) * 1000
            print(f"First Audio Latency (TTFA): {latency:.2f} ms")
        
        count += 1
    
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    throughput = count / total_time
    rtf = total_time / total_audio_duration if total_audio_duration > 0 else 0
    
    print("\n" + "="*50)
    print("STREAMING BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Sentences:      {count}")
    print(f"Total Audio Duration: {total_audio_duration:.2f} s")
    print(f"Total Wall Time:      {total_time:.2f} s")
    print(f"First Audio Latency:  {(first_audio_time - start_time)*1000:.2f} ms")
    print("-" * 50)
    print(f"RTF (Time/Audio):     {rtf:.4f} " + ("(Lower is better)" if rtf < 1.0 else "(Higher than real-time)"))
    print(f"Inverse RTF:          {1/rtf:.2f}x real-time")
    print(f"Avg Throughput:       {throughput:.2f} sentences/sec")
    print("="*50)

if __name__ == "__main__":
    benchmark_stream()
