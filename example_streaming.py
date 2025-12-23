"""
Example: Streaming TTS with Chatterbox

This script demonstrates real-time audio generation using the streaming API.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import soundfile as sf
from chatterbox.tts_stream import ChatterboxTTSStream


def main():
    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load streaming model
    model = ChatterboxTTSStream.from_pretrained(device=device)
    
    text = "Hello! This is a streaming test. Each chunk of audio is generated and yielded in real-time as tokens are produced."
    
    # Run 5 generations (first is warm-up)
    num_runs = 5
    
    for run in range(num_runs):
        is_warmup = (run == 0)
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{num_runs}" + (" (WARM-UP)" if is_warmup else ""))
        print(f"{'='*50}")
        
        all_chunks = []
        chunk_count = 0
        
        for audio_chunk, metrics in model.generate_stream(text, chunk_size=50):
            chunk_duration = audio_chunk.shape[0] / model.sr
            chunk_count += 1
            all_chunks.append(audio_chunk)
        
        # Concatenate and save
        full_audio = torch.cat(all_chunks, dim=0).numpy()
        output_filename = f"streaming_output_run{run + 1}.wav"
        sf.write(output_filename, full_audio, model.sr)
        print(f"Saved {output_filename}")


if __name__ == "__main__":
    main()
