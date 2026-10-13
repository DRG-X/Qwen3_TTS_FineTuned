"""Thin, behaviour-preserving wrapper around ``qwen_tts.Qwen3TTSModel`` for the Hindi fine-tune.

IMPORTANT - how Hindi works in this model (verified against qwen-tts==0.1.1 source):
  * Hindi is NOT a language id of the upstream model. ``language="Hindi"`` raises ValueError.
  * The fine-tune was trained with the ``language="auto"`` prompt layout (no language id in the
    codec prefix) and the NON-STREAMING text layout, with Devanagari text.
  * The voice is a single fixed speaker x-vector baked into codec-embedding slot 3000 and exposed
    as the custom-voice speaker ``hindi_female``.
This module therefore hard-codes ``language="auto"`` and ``non_streaming_mode=True`` and does not
expose a language argument, so the most common mistake (passing "Hindi") cannot happen.

The ``max_new_tokens`` heuristic is copied from ``training/htts_export.py::max_tokens_for``.
"""
from __future__ import annotations

import re
import warnings
from typing import Any

LANGUAGE = "auto"
NON_STREAMING_MODE = True
CUSTOM_SPEAKER_NAME = "hindi_female"

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def has_devanagari(text: str) -> bool:
    """True if ``text`` contains at least one Devanagari character."""
    return bool(_DEVANAGARI.search(text))


def default_max_new_tokens(text: str) -> int:
    """~1 codec frame per Hindi character with 2.5x headroom (same formula as training/htts_export.py)."""
    return int(len(text) * 2.5) + 60


class HindiTTS:
    """Convenience wrapper. ``tts`` must be a loaded ``qwen_tts.Qwen3TTSModel`` (custom_voice type)."""

    def __init__(self, tts: Any, speaker: str = CUSTOM_SPEAKER_NAME):
        self.tts = tts
        self.speaker = speaker

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        speaker: str = CUSTOM_SPEAKER_NAME,
    ) -> HindiTTS:
        import torch
        from qwen_tts import Qwen3TTSModel

        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        tts = Qwen3TTSModel.from_pretrained(
            model_id_or_path, device_map=device, dtype=torch_dtype, attn_implementation=attn_implementation
        )
        kind = getattr(tts.model, "tts_model_type", None)
        if kind != "custom_voice":
            raise ValueError(f"expected a custom_voice checkpoint, got tts_model_type={kind!r}")
        speakers = [s.lower() for s in (tts.get_supported_speakers() or [])]
        if speaker.lower() not in speakers:
            raise ValueError(f"speaker {speaker!r} not in checkpoint speakers {speakers}")
        return cls(tts, speaker)

    def synthesize(
        self,
        text: str,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        **generate_kwargs: Any,
    ) -> tuple[Any, int]:
        """Return ``(waveform_float32_numpy, sample_rate)``."""
        if not text or not text.strip():
            raise ValueError("text must be a non-empty string")
        if not has_devanagari(text):
            warnings.warn(
                "text contains no Devanagari characters; the model was fine-tuned on Devanagari Hindi only",
                stacklevel=2,
            )
        if seed is not None:
            import torch

            torch.manual_seed(seed)
        if max_new_tokens is None:
            max_new_tokens = default_max_new_tokens(text)
        wavs, sr = self.tts.generate_custom_voice(
            text=text,
            speaker=self.speaker,
            language=LANGUAGE,
            non_streaming_mode=NON_STREAMING_MODE,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return wavs[0], sr

    def synthesize_to_file(self, text: str, path: str, **kwargs: Any) -> str:
        import soundfile as sf

        wav, sr = self.synthesize(text, **kwargs)
        sf.write(path, wav, sr)
        return path
