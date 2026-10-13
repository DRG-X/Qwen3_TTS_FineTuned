"""The same call WITHOUT this repo's wrapper (identical to the original notebook, section 8)."""
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "<hf-username>/qwen3-tts-1.7b-hindi-female",  # or a local export directory
    device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa",
)
wavs, sr = tts.generate_custom_voice(
    text="नमस्ते, आप कैसे हैं?",
    speaker="hindi_female",
    language="auto",          # Hindi is not an upstream language id; training used the 'auto' layout
    non_streaming_mode=True,  # training used the non-streaming layout
    max_new_tokens=600,
)
sf.write("out.wav", wavs[0], sr)
