"""
Example: Chunked TTS Generation with Detailed Statistics

This demonstrates sentence-by-sentence generation with comprehensive metrics.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import soundfile as sf
import time
from chatterbox.tts import ChatterboxTTS


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for S3Gen")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Compile S3Gen: {args.compile}")
    print("="*80)
    
    model = ChatterboxTTS.from_pretrained(device=device, compile_s3gen=args.compile)
    
    text = """This is a chunked generation test, each sentence is processed separately. 
    This maintains optimal RTF while enabling streaming-like behavior.
    The audio quality remains high because we use full batch mode."""
    
    print(f"\nText to synthesize ({len(text)} chars):")
    print(f'"{text[:100]}..."')
    print("="*80)
    
    num_runs = 5
    run_stats = []
    
    for run in range(num_runs):
        is_warmup = (run == 0)
        
        print(f"\n{'#'*80}")
        print(f"# RUN {run + 1}/{num_runs}" + (" [WARM-UP]" if is_warmup else ""))
        print(f"{'#'*80}")
        
        run_start = time.perf_counter()
        
        all_audio = []
        sentence_stats = []
        
        print(f"\n{'Sentence':<8} {'RTF':>8} {'Duration':>10} {'GenTime':>10} {'Text':<40}")
        print("-"*80)
        
        for wav, metrics in model.generate_chunked(text, verbose=False):
            all_audio.append(wav.squeeze(0))
            sentence_stats.append(metrics)
            
            # Print per-sentence stats
            sentence_text = metrics['sentence'][:37] + "..." if len(metrics['sentence']) > 40 else metrics['sentence']
            print(f"{metrics['sentence_idx']:>8} {metrics['rtf']:>8.4f} {metrics['duration']:>9.2f}s {metrics['gen_time']:>9.2f}s  {sentence_text}")
        
        run_end = time.perf_counter()
        run_total_time = run_end - run_start
        
        # Calculate run statistics
        ttfa = sentence_stats[0]['gen_time']  # Time to First Audio = Gen time of 1st chunk
        total_audio = sum(s['duration'] for s in sentence_stats)
        total_gen = sum(s['gen_time'] for s in sentence_stats)
        overall_rtf = total_gen / total_audio
        avg_rtf = sum(s['rtf'] for s in sentence_stats) / len(sentence_stats)
        min_rtf = min(s['rtf'] for s in sentence_stats)
        max_rtf = max(s['rtf'] for s in sentence_stats)
        
        run_stats.append({
            'run': run + 1,
            'is_warmup': is_warmup,
            'ttfa': ttfa,
            'overall_rtf': overall_rtf,
            'avg_rtf': avg_rtf,
            'min_rtf': min_rtf,
            'max_rtf': max_rtf,
            'total_audio': total_audio,
            'total_gen': total_gen,
            'run_time': run_total_time,
            'num_sentences': len(sentence_stats),
        })
        
        print("-"*80)
        print(f"{'TOTAL':<8} {overall_rtf:>8.4f} {total_audio:>9.2f}s {total_gen:>9.2f}s")
        print(f"\n  Time to First Audio (TTFA): {ttfa*1000:.2f}ms")
        print(f"  Run Time: {run_total_time:.2f}s | Sentences: {len(sentence_stats)}")
        print(f"  RTF - Avg: {avg_rtf:.4f} | Min: {min_rtf:.4f} | Max: {max_rtf:.4f}")
        
        # Save only last run
        if run == num_runs - 1:
            full_audio = torch.cat(all_audio, dim=0).numpy()
            sf.write("chunked_output.wav", full_audio, model.sr)
            print(f"\n  Saved: chunked_output.wav")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY (excluding warm-up)")
    print("="*80)
    
    non_warmup = [s for s in run_stats if not s['is_warmup']]
    
    print(f"\n{'Run':<6} {'RTF':>10} {'TTFA':>10} {'Avg RTF':>10} {'Audio':>10} {'Time':>10}")
    print("-"*80)
    
    for s in run_stats:
        marker = " [W]" if s['is_warmup'] else ""
        print(f"{s['run']:<6}{marker} {s['overall_rtf']:>10.4f} {s['ttfa']*1000:>9.0f}ms {s['avg_rtf']:>10.4f} {s['total_audio']:>9.2f}s {s['run_time']:>9.2f}s")
    
    if non_warmup:
        print("-"*80)
        avg_overall_rtf = sum(s['overall_rtf'] for s in non_warmup) / len(non_warmup)
        avg_ttfa = sum(s['ttfa'] for s in non_warmup) / len(non_warmup)
        avg_time = sum(s['run_time'] for s in non_warmup) / len(non_warmup)
        avg_audio = sum(s['total_audio'] for s in non_warmup) / len(non_warmup)
        
        print(f"{'AVG':<6}  {avg_overall_rtf:>10.4f} {avg_ttfa*1000:>9.0f}ms {'-':>10} {avg_audio:>9.2f}s {avg_time:>9.2f}s")
        print(f"\n  Average RTF (non-warmup):    {avg_overall_rtf:.4f}")
        print(f"  Average TTFA (non-warmup):   {avg_ttfa*1000:.2f}ms")
        print(f"  Average Audio Duration:      {avg_audio:.2f}s")
        print(f"  Average Generation Time:     {avg_time:.2f}s")


if __name__ == "__main__":
    main()

