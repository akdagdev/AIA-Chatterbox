#!/usr/bin/env python3
"""
Test script for batch tokenization in Chatterbox TTS.
Run from project root: python tests/test_batch_tokenizer.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch


def test_mtl_tokenizer_batch():
    """Test MTLTokenizer batch tokenization."""
    from chatterbox.models.tokenizers import MTLTokenizer
    from huggingface_hub import snapshot_download
    import os
    
    # Download tokenizer vocab if needed
    ckpt_dir = Path(snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        allow_patterns=["grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"],
        token=os.getenv("HF_TOKEN"),
    ))
    
    tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
    
    # Test texts with different lengths
    texts = [
        "Hello world.",
        "This is a longer test sentence for batching.",
        "Short.",
    ]
    
    print("Testing MTLTokenizer.text_to_tokens_batch()...")
    
    # Test batch tokenization
    tokens, mask = tokenizer.text_to_tokens_batch(texts, language_id="en")
    
    print(f"  Input texts: {len(texts)}")
    print(f"  Output tokens shape: {tokens.shape}")
    print(f"  Attention mask shape: {mask.shape}")
    
    # Assertions
    assert tokens.shape[0] == len(texts), f"Batch size mismatch: {tokens.shape[0]} != {len(texts)}"
    assert mask.shape == tokens.shape, f"Mask shape mismatch"
    
    # Verify mask sums match actual token counts
    for i, text in enumerate(texts):
        single_tokens = tokenizer.text_to_tokens(text, language_id="en")
        single_len = single_tokens.shape[1]
        mask_sum = int(mask[i].sum().item())
        print(f"  Text {i}: '{text[:30]}...' -> single_len={single_len}, mask_sum={mask_sum}")
        assert single_len == mask_sum, f"Length mismatch for text {i}"
    
    # Verify tokens match between single and batch
    for i, text in enumerate(texts):
        single_tokens = tokenizer.text_to_tokens(text, language_id="en").squeeze(0)
        batch_tokens = tokens[i, :len(single_tokens)]
        assert torch.equal(single_tokens, batch_tokens), f"Token mismatch for text {i}"
    
    print("  ✓ All assertions passed!")
    return True


def test_empty_batch():
    """Test error handling for empty batch."""
    from chatterbox.models.tokenizers import MTLTokenizer
    from huggingface_hub import snapshot_download
    import os
    
    ckpt_dir = Path(snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        allow_patterns=["grapheme_mtl_merged_expanded_v1.json", "Cangjie5_TC.json"],
        token=os.getenv("HF_TOKEN"),
    ))
    
    tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
    
    print("Testing empty batch handling...")
    try:
        tokenizer.text_to_tokens_batch([], language_id="en")
        print("  ✗ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Chatterbox Batch Tokenizer Test")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 2
    
    try:
        if test_mtl_tokenizer_batch():
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test failed with error: {e}")
    
    print()
    
    try:
        if test_empty_batch():
            tests_passed += 1
    except Exception as e:
        print(f"  ✗ Test failed with error: {e}")
    
    print()
    print("=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    sys.exit(0 if tests_passed == tests_total else 1)
