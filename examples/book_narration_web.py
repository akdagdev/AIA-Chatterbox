
import os
import re
import io
import argparse
import logging
import tempfile
from typing import List, Generator

import torch
import torchaudio
import soundfile as sf
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

# Adjust path to find chatterbox modules
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SpeechRequest

# Constants
SAMPLE_RATE = 24000 

app = FastAPI(title="Chatterbox Book Narration")
logger = logging.getLogger("book_narration")
logging.basicConfig(level=logging.INFO)

# Global model holder
model: ChatterboxMultilingualTTS = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex, preserving punctuation.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def batch_generator(items: List[any], batch_size: int) -> Generator[List[any], None, None]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

def load_models(checkpoint_path: str = None):
    global model, device
    
    logger.info(f"Loading ChatterboxMultilingualTTS on {device}...")
    
    if checkpoint_path:
        logger.info(f"Loading from local checkpoint: {checkpoint_path}")
        model = ChatterboxMultilingualTTS.from_local(checkpoint_path, device)
    else:
        logger.info("Loading from pretrained (HuggingFace)...")
        model = ChatterboxMultilingualTTS.from_pretrained(device)
        
    logger.info("Model loaded successfully.")

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chatterbox Book Narrator</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; background-color: #f4f4f9; color: #333; }
            h1 { color: #2c3e50; text-align: center; }
            .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            textarea { width: 100%; height: 200px; margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; resize: vertical; box-sizing: border-box; }
            .form-group { margin-bottom: 20px; }
            label { display: block; margin-bottom: 8px; font-weight: 600; }
            input[type="file"] { padding: 10px; background: #f8f9fa; width: 100%; border-radius: 4px; box-sizing: border-box; }
            button { width: 100%; padding: 15px; cursor: pointer; background-color: #007bff; color: white; border: none; border-radius: 4px; font-size: 16px; font-weight: bold; transition: background 0.3s; }
            button:hover { background-color: #0056b3; }
            #loading { display: none; text-align: center; margin-top: 20px; color: #666; }
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 2s linear infinite; display: inline-block; vertical-align: middle; margin-right: 10px; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Book Narrator</h1>
            <form action="/generate" method="post" enctype="multipart/form-data" onsubmit="document.getElementById('loading').style.display='block'">
                <div class="form-group">
                    <label for="text">Book Content:</label>
                    <textarea name="text" id="text" required placeholder="Paste the text you want to narrate here..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="ref_audio">Voice Reference (WAV/MP3):</label>
                    <input type="file" name="ref_audio" id="ref_audio" accept="audio/*" required>
                </div>
                
                <button type="submit">Generate Narration</button>
            </form>
            <div id="loading">
                <div class="spinner"></div> Generating audio... Please wait.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/generate")
async def generate(text: str = Form(...), ref_audio: UploadFile = File(...)):
    global model, device
    
    if not text:
        return Response("No text provided", status_code=400)
    
    logger.info(f"Received request. Text length: {len(text)} chars")
    
    # Save UploadFile to a temporary file because get_conditioning_for_prompt expects a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(ref_audio.filename).suffix) as tmp_ref:
        tmp_ref.write(await ref_audio.read())
        tmp_ref_path = tmp_ref.name
    
    try:
        # Pre-compute conditioning once for the reference audio
        # Logic matches benchmark_batch.py: create conditional cache or handle via prompt path
        # Since we are splitting text, we have multiple segments.
        # We can either pass the same path to all, or pre-compute.
        # Let's pass the path to generate_batch and let logic handle it (it caches internally if optimized, or we use the cached object).
        # Actually generate_batch optimization in mtl_tts.py lines 446+ handles simple caching for same path.
        
        sentences = split_sentences(text)
        logger.info(f"Split text into {len(sentences)} sentences.")
        
        BATCH_SIZE = 4 # Adjust as needed
        all_audio_segments = []
        
        for batch_texts in batch_generator(sentences, BATCH_SIZE):
            # Generate batch
            # ChatterboxMultilingualTTS.generate_batch returns List[Tensor]
            # Assumes 'en' language for now, or could detect.
            # Using audio_prompt_path=tmp_ref_path for all items in batch
            
            # Create list of prompts (same file for all)
            prompts = [tmp_ref_path] * len(batch_texts)
            
            with torch.no_grad():
                audios = model.generate_batch(
                    texts=batch_texts,
                    language_id="en", # Defaulting to English
                    audio_prompt_path=prompts
                )
            
            for audio in audios:
                # Audio is tensor [1, T] or [C, T]
                if audio.ndim == 2:
                    audio = audio.cpu()
                else:
                    audio = audio.unsqueeze(0).cpu()
                
                all_audio_segments.append(audio)
                
                # Add small silence (0.2s)
                silence_len = int(0.2 * model.sr)
                silence = torch.zeros(1, silence_len)
                all_audio_segments.append(silence)
        
        # Concatenate
        if all_audio_segments:
            final_audio = torch.cat(all_audio_segments, dim=1)
        else:
            final_audio = torch.zeros(1, 1) # Fallback

        # Convert to bytes
        buffer = io.BytesIO()
        # torchaudio.save fails with BytesIO on some versions/backends (torchcodec)
        # Use soundfile instead
        wav_numpy = final_audio.squeeze(0).cpu().numpy()
        sf.write(buffer, wav_numpy, model.sr, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=narration.wav"})
        
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_ref_path):
            os.remove(tmp_ref_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint directory (optional). If not provided, downloads from HF.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    # Load model
    load_models(args.checkpoint)
    
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
