import os
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download


@pytest.fixture(scope="session")
def ckpt_dir():
    """Download and cache model checkpoint directory (shared across all tests)."""
    return Path(snapshot_download(
        repo_id="ResembleAI/chatterbox",
        repo_type="model",
        allow_patterns=[
            "grapheme_mtl_merged_expanded_v1.json",
            "Cangjie5_TC.json",
        ],
        token=os.getenv("HF_TOKEN"),
    ))


@pytest.fixture(scope="session")
def tokenizer(ckpt_dir):
    """Shared MTLTokenizer instance."""
    from chatterbox.models.tokenizers import MTLTokenizer
    return MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
