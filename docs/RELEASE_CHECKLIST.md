# Release checklist

Legend: [x] done in this repo, [ ] still to do (cannot be done without GPU/network/accounts).

## A. Blocking
- [ ] Resolve dataset licence/voice-likeness questions (docs/LICENSING.md #2, #3). Keep the founder's approval in writing.
- [ ] Re-export on a GPU machine from `adapter_best.pt` + `speaker_embedding.pt` + `test_sentences.jsonl` using the patched `training/htts_export.py` (the original export crashed).
- [ ] `python scripts/verify_release_dir.py --base <base snapshot> --export <export dir>` -> `OK`.
- [ ] Reload from disk and synthesise (`examples/synthesize.py`). Listen to every file. If unintelligible, **stop** and investigate (merge precision, speaker slot, prompt layout) before publishing anything.
- [ ] Whisper CER on the generated files with `jiwer` cross-check; record the numbers *as measured*, with sentence count and the validation-set caveat.
- [ ] Compare merged-model output with adapter-on-base output on one sentence (merge-equivalence in bf16, not just fp32).
- [ ] Compare `model.safetensors` key set with the base (minus `speaker_encoder.*`); record tensor count.

## B. GitHub repository
- [x] README, CONTRIBUTING, CITATION.cff, LICENSE (Apache-2.0), NOTICE, .gitignore, .gitattributes, CI workflow (CI itself untested), tests (21 pass locally).
- [ ] Replace `<github-username>` / `<hf-username>` placeholders; fill CITATION.cff date/affiliation.
- [ ] `git init`; confirm `git status` shows no `*.pt`, `*.safetensors`, `*.wav`, dataset files, tokens. Run `git ls-files | xargs du -ch | tail -1` (expect < 5 MB).
- [ ] Push, enable Actions, confirm CI green; tag `v0.1.0`; create a GitHub release linking the HF model.

## C. Hugging Face model repo (only after A)
- [ ] `huggingface-cli login`; `huggingface-cli repo create qwen3-tts-1.7b-hindi-female --type model` (start **private**).
- [ ] Upload the verified export dir (`model.safetensors` etc.), `model_card/README.md` as `README.md`, `adapter/` (safetensors + `adapter_config.json`), 3-5 *generated* sample WAVs with their text. Never upload `lora_adapter.pt`/`speaker_embedding.pt` pickles.
- [ ] Fill model-card TODOs: licence, real metrics, sample table; remove the "UNVERIFIED" banner only when section A is complete.
- [ ] Large files: `huggingface_hub` uploads via LFS/Xet automatically; do not commit weights to GitHub.
- [ ] Versioning: tag the HF repo `v0.1.0`; pin `revision="v0.1.0"` in the README quickstart.

## D. Post-release validation (fresh environment)
- [ ] New venv + `pip install -r requirements-inference.txt`; run the README quickstart against the *published* repo id; compare output to the pre-release samples.
- [ ] Confirm model card renders, licence badge correct, links work, private -> public switch last.
