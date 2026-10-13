"""Builds a tiny RANDOMLY-INITIALISED Qwen3-TTS talker with the same module structure as the 1.7B model.

Used only for CPU unit tests of the loss/alignment code. Requires qwen-tts==0.1.1 + transformers==4.57.3.
"""
import types

import torch

NUM_GROUPS = 4  # real model: 16


def build_tiny_core(seed=0):
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig, Qwen3TTSTalkerConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerForConditionalGeneration

    torch.manual_seed(seed)
    cp = Qwen3TTSTalkerCodePredictorConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_code_groups=NUM_GROUPS, max_position_embeddings=64,
    )
    tc = Qwen3TTSTalkerConfig(
        code_predictor_config=cp, vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, num_code_groups=NUM_GROUPS, text_hidden_size=32,
        codec_eos_token_id=100, codec_think_id=101, codec_nothink_id=102, codec_think_bos_id=103,
        codec_think_eos_id=104, codec_pad_id=105, codec_bos_id=106, max_position_embeddings=256,
        text_vocab_size=200,
        # structure of the real config: multimodal RoPE; sections must sum to head_dim/2 (=8 here)
        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "interleaved": True},
    )
    talker = Qwen3TTSTalkerForConditionalGeneration(tc).float().eval()
    hf_cfg = types.SimpleNamespace(
        talker_config=tc, tts_pad_token_id=190, tts_bos_token_id=191, tts_eos_token_id=192,
        im_start_token_id=193, im_end_token_id=194,
    )
    core = types.SimpleNamespace(talker=talker, config=hf_cfg)
    return core, hf_cfg


def make_batch(hf_cfg, texts=((7, 8, 9, 10, 11), (7, 8, 9, 12, 13, 14, 15)), frames=(5, 3), seed=1):
    """texts: token lists WITH 3 leading role tokens (like tokenize_for_training output)."""
    from htts_common import TTSCollator

    g = torch.Generator().manual_seed(seed)
    rows = []
    for t, n in zip(texts, frames):
        codes = torch.randint(0, 64, (n, NUM_GROUPS), generator=g)
        rows.append({"text_ids": torch.tensor(t), "audio_codes": codes})
    return TTSCollator(hf_cfg)(rows), rows
