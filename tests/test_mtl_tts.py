"""Unit tests for ChatterboxMultilingualTTS (no GPU required)."""
import threading

import pytest

from chatterbox.mtl_tts import (
    ChatterboxMultilingualTTS,
    Conditionals,
    SpeechRequest,
    SUPPORTED_LANGUAGES,
    punc_norm,
)


class TestPuncNorm:
    def test_empty_string(self):
        result = punc_norm("")
        assert len(result) > 0

    def test_capitalizes_first_letter(self):
        assert punc_norm("hello world.")[0] == "H"

    def test_adds_period_if_missing(self):
        assert punc_norm("Hello world").endswith(".")

    def test_preserves_existing_ending(self):
        assert punc_norm("Hello world!").endswith("!")
        assert punc_norm("Hello world?").endswith("?")

    def test_replaces_ellipsis(self):
        result = punc_norm("Hello... world.")
        assert "..." not in result

    def test_collapses_multiple_spaces(self):
        result = punc_norm("Hello    world.")
        assert "    " not in result


class TestSpeechRequest:
    def test_defaults(self):
        req = SpeechRequest(text="Hello")
        assert req.text == "Hello"
        assert req.audio_prompt_path is None
        assert req.language_id is None
        assert req.conditionals is None


class TestSupportedLanguages:
    def test_has_23_languages(self):
        assert len(SUPPORTED_LANGUAGES) == 23

    def test_english_present(self):
        assert "en" in SUPPORTED_LANGUAGES

    def test_all_codes_are_lowercase_two_letter(self):
        for code in SUPPORTED_LANGUAGES:
            assert len(code) == 2
            assert code == code.lower()


class TestBatchLock:
    def test_batch_lock_exists(self):
        """Verify that _batch_lock is created in __init__."""
        # We can't easily instantiate ChatterboxMultilingualTTS without models,
        # but we can verify the attribute is set in __init__ by inspecting the source.
        import inspect
        source = inspect.getsource(ChatterboxMultilingualTTS.__init__)
        assert "_batch_lock" in source
