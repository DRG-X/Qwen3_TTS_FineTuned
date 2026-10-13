# Evaluation - what exists and what does not

## Measured (from the real training run; `docs/train_metrics.jsonl`, `docs/train_summary.json`, `docs/training_curves.png`)
Teacher-forced metrics on the 112-utterance validation split (9,582 tokens). Best checkpoint = step 500.

| | step 0 (base) | step 500 (released adapter) | step 750 (final) |
|---|---|---|---|
| codebook-0 CE / top-1 | 1.366 / 0.600 | **0.883 / 0.694** | 0.911 / 0.693 |
| sub-talker CE / top-1 | 5.038 / 0.105 | **4.451 / 0.148** | 4.449 / 0.148 |
| total (0.3 weight on sub) | - | **2.218** | 2.246 |

Training stopped early at step 750 (of 1,038 planned), 5.28 h on 2x T4, ~230 tokens/s. Validation loss flattens by ~step 250 while
train loss keeps falling (train codebook-0 CE ~0.63 vs val ~0.91 at step 750): the model overfits a single-speaker, ~10 h corpus.

**These numbers do not measure intelligibility or naturalness.** Teacher-forced CE/accuracy can look healthy for unintelligible audio.

## Not measured (do not claim)
* Whisper CER/WER of generated speech. The notebook's evaluation cell never ran because export crashed (docs/AUDIT.md #1).
* MOS / listening tests, speaker-similarity, latency/RTF, memory use, long-text behaviour, other hardware, other torch versions.
* Any comparison with the base model's audio (the 13 base-model samples were written on Kaggle but are not in this repo).

## Caveats for any future evaluation
* The 8 listening sentences were drawn from the validation split used for checkpoint selection and early stopping -> not a held-out test.
* Utterance-level split; sentence overlap between train and val has not been checked.
* Whisper CER on Hindi is noisy and the notebook's CER function is hand-written (cross-check with `jiwer`).
* With ~13 sentences, report results as a smoke test, not a benchmark.

## What was tested in this repository (CPU, no weights)
`pytest -q tests` -> **21 passed** (torch 2.14.1 CPU, transformers 4.57.3, qwen-tts 0.1.1, Python 3.12): config keys, collator layout,
LoRA injection/merge/round-trip, config transform, speaker-slot write, export-copy regression, inference-wrapper contract, loss alignment on a tiny
random model, and the real `adapter_best.pt` schema (466 tensors, 36,028,416 params, all finite).
The training environment was torch 2.10.0+cu128 / Python 3.12 (Kaggle); the tests ran on a different torch.

## Untested - exact commands
```bash
# GPU: re-export from saved artifacts, reload from disk, synthesise
CUDA_VISIBLE_DEVICES=0 HTTS_CONFIG=configs/train_hindi_female.local.json python training/htts_export.py
python scripts/verify_release_dir.py --base <base snapshot dir> --export work/qwen3tts-1.7b-hindi-female
python examples/synthesize.py --model work/qwen3tts-1.7b-hindi-female
```
