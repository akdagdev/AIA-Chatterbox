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
    
    count = 0
    for i, audio in enumerate(stream):
        if i == 0:
            if device == "cuda":
                torch.cuda.synchronize() # Ensure it's truly done for timing
            first_audio_time = time.time()
            latency = (first_audio_time - start_time) * 1000
            print(f"First Audio Latency (TTFA): {latency:.2f} ms")
        
        count += 1
        # Emulate network send
        # time.sleep(0.01) 
    
    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time
    throughput = count / total_time
    
    print(f"\nTotal Sentences: {count}")
    print(f"Total Time:      {total_time:.2f} s")
    print(f"Avg Throughput:  {throughput:.2f} sentences/sec")

if __name__ == "__main__":
    benchmark_stream()
