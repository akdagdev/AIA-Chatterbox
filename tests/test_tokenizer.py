"""Tests for MTLTokenizer batch tokenization."""
import pytest
import torch


def test_batch_tokenization_shape(tokenizer):
    texts = [
        "Hello world.",
        "This is a longer test sentence for batching.",
        "Short.",
    ]
    tokens, mask = tokenizer.text_to_tokens_batch(texts, language_id="en")

    assert tokens.shape[0] == len(texts)
    assert mask.shape == tokens.shape


def test_batch_matches_single(tokenizer):
    texts = [
        "Hello world.",
        "This is a longer test sentence for batching.",
        "Short.",
    ]
    tokens, mask = tokenizer.text_to_tokens_batch(texts, language_id="en")

    for i, text in enumerate(texts):
        single_tokens = tokenizer.text_to_tokens(text, language_id="en")
        single_len = single_tokens.shape[1]
        mask_sum = int(mask[i].sum().item())
        assert single_len == mask_sum, f"Length mismatch for text {i}"

        batch_tokens = tokens[i, :single_len]
        assert torch.equal(single_tokens.squeeze(0), batch_tokens), f"Token mismatch for text {i}"


def test_empty_batch_raises(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.text_to_tokens_batch([], language_id="en")


def test_batch_with_sot_eot(tokenizer):
    texts = ["Test sentence."]
    sot, eot = 255, 0
    tokens, mask = tokenizer.text_to_tokens_batch(
        texts, language_id="en", sot_token=sot, eot_token=eot
    )
    assert tokens[0, 0].item() == sot
    last_valid = int(mask[0].sum().item()) - 1
    assert tokens[0, last_valid].item() == eot


@pytest.mark.parametrize("lang", ["en", "tr", "fr", "de", "es"])
def test_batch_multilingual(tokenizer, lang):
    texts = ["Hello.", "Test."]
    tokens, mask = tokenizer.text_to_tokens_batch(texts, language_id=lang)
    assert tokens.shape[0] == 2
    assert mask.sum() > 0
