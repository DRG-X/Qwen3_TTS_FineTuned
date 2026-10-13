# Audit of the source notebook (Phase A)

Source of truth: the Kaggle notebook `notebook86bf2c5571.ipynb` (26 cells) plus the artifacts supplied from the same run.
Library behaviour was checked against the *source* of `qwen-tts==0.1.1` and `transformers==4.57.3`.

## Findings

| # | Finding | Evidence | Resolution in this repo |
|---|---|---|---|
| 1 | **The notebook's export crashed**; the exported folder was never reloaded, so no fine-tuned audio, no Whisper CER and no `eval_report.json` exist. | Cell 20 output: `AssertionError` while reloading. | Not fixable by documentation: the model must be re-exported and validated (docs/RELEASE_CHECKLIST.md). |
| 2 | Root cause: `shutil.ignore_patterns("model.safetensors", ...)` matches at *every* depth and also dropped `speech_tokenizer/model.safetensors`. | Reproduced on a mock tree; regression test `test_original_copytree_pattern_dropped_codec_weights`. | `htts_common.copy_base_assets` skips weights only at top level; `verify_export_dir` runs automatically after export (patches P1, P2). |
| 3 | Hindi is not an upstream language id. `language="Hindi"` raises `ValueError`; training used the `auto` layout. | `get_supported_languages` in the log; `_validate_languages` in qwen-tts source. | `HindiTTS` hard-codes `language="auto"`, `non_streaming_mode=True`. |
| 4 | `model.safetensors` alone is not a loadable checkpoint. | `from_pretrained` needs `config.json`, `generation_config.json`, `speech_tokenizer/` (config + weights), tokenizer/processor files. | Export copies every non-weight file from the base snapshot; verifier compares against that snapshot. |
| 5 | Data volume stated as "~5.2 h" in notebook prose; the executed log shows **10.46 h** (5,648 utt). | Cell 15 output. | Docs use the logged numbers. |
| 6 | The 8 "held-out" listening sentences come from the validation split that also selected the checkpoint and drove early stopping; splits are by utterance (text overlap not checked). | `htts_prepare_data.py`, `htts_train.py`. | Documented as a limitation; any CER on them is a smoke test, not a benchmark. |
| 7 | LoRA delta is merged into bf16 weights; merged-vs-adapter equivalence was never tested. | `merge_lora` code. | fp32 equivalence test passes; bf16 rounding bound measured in tests. End-to-end merged audio is still unverified. |
| 8 | Adapter/speaker artifacts are pickles. | `torch.save` in training/export. | `scripts/convert_artifacts_to_safetensors.py` (round-trip asserted). |
| 9 | The time guard is disabled when `notebook_start_ts` is `null` (it falls back to "now" on every call). | `seconds_left` logic. | Documented in docs/TRAINING.md. |
| 10 | Kaggle metadata says `isGpuEnabled: False` although the log shows 2x T4. | Notebook metadata vs. log. | Ignored (metadata artifact). |

## Verified claims about the custom loss (see tests/test_loss_alignment.py)
* Main loss equals the library's loss when the library gets *unshifted* labels (one shift).
* The legacy pattern (manual shift + library shift) equals a two-steps-ahead CE.
* `forward_sub_talker_finetune` grades column *j* against codebook *j+2*; the library cannot express the correct alignment for the last codebook; `compute_loss` covers all 15 columns.
* Frame *p* is predicted from the hidden state at position *p-1*.
* On the real base model the step-0 diagnostic logged a corrected codebook-0 CE of 1.33 vs 10.89 for the legacy loss.

## Not verified (no weights / GPU / network access during this audit)
* Loading the re-exported model, generating audio, ASR/CER, any listening test.
* Tensor count/key set of the exported `model.safetensors` against the base (the original log reports 404 tensors).
* Claims in the notebook about the upstream `sft_12hz.py` and earlier runs (not supplied).
