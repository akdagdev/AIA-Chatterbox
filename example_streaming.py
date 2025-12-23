"""
Example: Streaming TTS with Chatterbox

This script demonstrates real-time audio generation using the streaming API.
"""
import torch
import torchaudio as ta
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
    
    # Collect all chunks
    all_chunks = []
    chunk_count = 0
    
    print("Generating audio chunks...")
    # generate_stream includes verbose=True by default for metrics printing
    for audio_chunk, metrics in model.generate_stream(text, chunk_size=50):
        chunk_duration = audio_chunk.shape[0] / model.sr
        chunk_count += 1
        print(f"  Chunk {chunk_count}: {audio_chunk.shape[0]} samples ({chunk_duration:.2f}s)")
        all_chunks.append(audio_chunk)
    
    # Concatenate and save
    full_audio = torch.cat(all_chunks, dim=0).unsqueeze(0)
    ta.save("streaming_output.wav", full_audio, model.sr)
    print(f"Saved streaming_output.wav")


if __name__ == "__main__":
    main()
