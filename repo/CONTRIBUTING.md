# Contributing

Thanks for helping. Please keep the project honest: **no unmeasured claims** in code, docs or model card.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch qwen-tts==0.1.1 safetensors pytest   # CPU is enough for the test suite
pip install -e . --no-deps
pytest -q tests                                        # 21 tests, ~6 s, no weights needed
```
Optional: `HINDI_ADAPTER_PT=/path/adapter_best.pt pytest tests/test_smoke.py::test_real_adapter_schema`.

## Ground rules
* `training/` holds the original notebook scripts verbatim. Changes there need a `[release-patch Pn]` marker, a note in `docs/AUDIT.md`, and a test.
* Do not change `compute_loss`/`TTSCollator` without extending `tests/test_loss_alignment.py` to demonstrate equivalence or correctness.
* Never commit weights, datasets, audio, tokens. Large files belong on Hugging Face.
* Label anything you did not run as UNVERIFIED and include the exact command.
* Good first contributions: a GPU smoke test (marked `@pytest.mark.gpu`), jiwer-based CER script, long-text chunking, other-speaker recipe, merge-precision study (bf16 vs fp32).

## Pull requests
Small, focused, with tests; describe hardware/versions used for any manual verification.
