"""Hindi (female) fine-tune of Qwen3-TTS-12Hz-1.7B-Base: inference helpers.

Heavy dependencies (torch, qwen_tts) are imported lazily so that
``import qwen3_tts_hindi`` works on a machine without a GPU stack.
"""
from .inference import (
    CUSTOM_SPEAKER_NAME,
    LANGUAGE,
    NON_STREAMING_MODE,
    HindiTTS,
    default_max_new_tokens,
    has_devanagari,
)

__all__ = [
    "CUSTOM_SPEAKER_NAME",
    "LANGUAGE",
    "NON_STREAMING_MODE",
    "HindiTTS",
    "default_max_new_tokens",
    "has_devanagari",
]
__version__ = "0.1.0.dev0"
