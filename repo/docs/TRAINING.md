# Training, data preparation and export

All code in `training/` is extracted **verbatim** from the notebook's `%%writefile` cells. The only changes are the two
marked `[release-patch P1/P2]` blocks (export fix + verification); `htts_common.py` lines 1-462 are byte-identical to the original.

## Pipeline

| Stage | Script | Notes |
|---|---|---|
| 1. Data prep | `htts_prepare_data.py` | Streams `SPRINGLab/IndicTTS-Hindi`, keeps female rows, 24 kHz mono, trims silence (`trim_top_db=35`), keeps 1-15 s, extracts 12 Hz codec codes with `Qwen/Qwen3-TTS-Tokenizer-12Hz`, tokenises text with the model's processor (with structural assertions), computes one fixed speaker x-vector (mean over 40 clips), splits train/val/listening-test. Logged: 5,983 female utterances -> 5,648 after filtering (10.46 h); 5,529 train (10.23 h) / 112 val / 8 listening; 7 more dropped by `max_seq_len=400`. |
| 2. Train | `htts_train.py` | DDP LoRA (notebook: 2x T4). Effective batch 32 (8/GPU x 2 GPUs x 2 accumulation). AdamW (betas 0.9/0.99, wd 0), lr 1e-4, cosine to 10 %, 31-step warm-up, grad-clip 1.0, 6 epochs planned (1,038 steps), eval every 50 steps, early stop after 5 evals without improvement. |
| 3. Export | `htts_export.py` | Merges `adapter_best.pt`, writes a standalone `custom_voice` checkpoint, reloads it, synthesises, (optionally) Whisper-CER. |

Hyper-parameters live in `configs/train_hindi_female.original.json` (Kaggle paths, exactly as run) and
`configs/train_hindi_female.local.json` (same values, paths under `./work`). Select with `HTTS_CONFIG=...`.

Run everything: `bash scripts/run_pipeline.sh` (unverified outside Kaggle).

### Time guard
`seconds_left()` uses `notebook_start_ts`. With `null` it falls back to the current time on every call, i.e. the guard never
fires. Set a real epoch timestamp if you need `max_total_hours` to take effect.

## LoRA design (custom, not PEFT)
* Targets (233 layers, 36,028,416 trainable parameters = 1.85 % of 1,953 M): talker q/k/v/o/gate/up/down (196 layers, r=32, alpha=32),
  `text_projection.linear_fc1/2` (2 layers, r=32), code-predictor q/k/v/o/gate/up/down (35 layers, r=8, alpha=8). Dropout 0.05.
* Adapter keys look like `talker.model.layers.0.self_attn.q_proj.lora_A`. Use `inject_lora` + `load_lora_state_dict` from `training/htts_common.py`.
* Export merges the delta into the (bf16) base weights; `speaker_encoder` is dropped and the fixed x-vector is written into
  codec-embedding row 3000 with `spk_id={"hindi_female": 3000}`, `spk_is_dialect=False`, `tts_model_type="custom_voice"`.

## Technical contributions
1. **Loss alignment fix.** Upstream's helper functions apply HF's internal label shift on top of a manual shift (talker) and
   grade sub-talker column *j* against codebook *j+2*. `compute_loss` shifts exactly once and grades every codebook. Demonstrated in
   `tests/test_loss_alignment.py` (tiny random model) and by the step-0 diagnostic on the real model (corrected CE 1.33 vs legacy 10.89).
2. **Hindi without a language id.** Fine-tuning uses the `auto` / non-streaming layout so no upstream vocabulary change is needed;
   layout parity with `generate()` is asserted by `test_collator_layout_matches_documented_layout`.
3. **Fail-fast data and export validation** (processor-structure assertions, export manifest check).

## Reproducing the release from the saved adapter (no retraining)
Place `adapter_best.pt`, `speaker_embedding.pt`, `test_sentences.jsonl` in `output_dir`, then
`CUDA_VISIBLE_DEVICES=0 python training/htts_export.py` (GPU + Hugging Face access required). Unverified.
