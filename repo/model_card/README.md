---
language: [hi]
pipeline_tag: text-to-speech
base_model: Qwen/Qwen3-TTS-12Hz-1.7B-Base
datasets: [SPRINGLab/IndicTTS-Hindi]
library_name: qwen-tts
tags: [text-to-speech, hindi, lora, qwen3-tts]
license: other
license_name: pending-verification   # TODO: set after docs/LICENSING.md items 1-4 are resolved
---

> **UNVERIFIED DRAFT - do not publish until docs/RELEASE_CHECKLIST.md section A is complete.**
> The exported model had not been reloaded or used to synthesise audio when this card was written.

# Qwen3-TTS-1.7B Hindi (female) - LoRA fine-tune

Hindi text-to-speech with a single fixed female voice, fine-tuned from `Qwen/Qwen3-TTS-12Hz-1.7B-Base` with LoRA (talker, text projection, code predictor) and merged into a standalone `custom_voice` checkpoint.
Code, training scripts, tests: https://github.com/<github-username>/qwen3-tts-hindi

## Usage
```bash
pip install qwen-tts==0.1.1 soundfile
```
```python
import soundfile as sf, torch
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained("<hf-username>/qwen3-tts-1.7b-hindi-female", device_map="cuda:0",
                                    dtype=torch.bfloat16, attn_implementation="sdpa")
wavs, sr = tts.generate_custom_voice(text="नमस्ते, आप कैसे हैं?", speaker="hindi_female",
                                     language="auto", non_streaming_mode=True, max_new_tokens=600)
sf.write("out.wav", wavs[0], sr)
```
**Hindi is not an upstream language id.** Use `language="auto"` and `non_streaming_mode=True` (the layouts used in training); `language="Hindi"` raises an error.
Input should be Devanagari. Speaker: `hindi_female` (fixed x-vector baked into codec-embedding slot 3000). Required files (all in this repo): `config.json`, `generation_config.json`, `model.safetensors`, tokenizer/processor files, and `speech_tokenizer/` (config **and** weights).

## Training data and procedure
* Data: female utterances of `SPRINGLab/IndicTTS-Hindi` (derived from the IIT Madras IndicTTS database): 5,983 -> 5,648 after filtering to 1-15 s (10.46 h); 5,529 train (10.23 h) / 112 validation / 8 listening-test.
* LoRA: r=32 (talker, text projection), r=8 (code predictor), 36.0 M trainable parameters; AdamW lr 1e-4 cosine, effective batch 32, 2x T4, early-stopped at step 750 of 1,038; released checkpoint = step 500 (lowest validation loss).
* Custom loss alignment (single shift, all 15 sub-codebooks) instead of the library helper path - see the GitHub repo for tests.

## Evaluation
Teacher-forced validation (112 utterances), step 500: codebook-0 CE 0.883 / top-1 0.694; sub-talker CE 4.451 / top-1 0.148.
**No intelligibility (CER/WER), naturalness (MOS), speaker-similarity or latency evaluation has been run.** <!-- TODO: add measured CER with sentence count and caveats -->
The validation split also selected the checkpoint, so it is not an unbiased test set.

## Samples
<!-- TODO: table of generated WAVs + text, added only after listening to them -->

## Limitations and risks
Single speaker; Hindi (Devanagari) only; trained on clips <=15 s so long inputs are untested; may mispronounce names, numbers, English words; overfits after ~250 steps; outputs not verified for intelligibility.
The voice is derived from a real speaker in the training corpus - do not use it to impersonate people or to deceive.

## License
<!-- TODO -->Base model: Apache-2.0 (verify). Dataset terms: see IIT Madras IndicTTS licence (verify). Weights licence: pending. Attribution: Qwen3-TTS (Qwen team, Alibaba Cloud), SPRING Lab / IIT Madras IndicTTS.

## Citation
See `CITATION.cff` in the GitHub repo.
