# Qwen3-TTS Hindi Fine-Tune

LoRA fine-tuning of **Qwen3-TTS-12Hz-1.7B-Base** for **Hindi female speech**, with the training pipeline, corrected loss alignment, inference helpers, export utilities, and CPU tests.

> **Status: Exported model tested.** The exported checkpoint has been loaded locally and used to synthesize multiple Hindi sentences. The generated audio was listened to by the author, confirming that Hindi speech generation works in the tested environment. Formal evaluations of intelligibility, naturalness, and pronunciation accuracy have not yet been completed.
>
> Model weights are available on [Hugging Face](https://huggingface.co/DRG-X/qwen3_TTS_FineTuned-Hindi). See `docs/EVALUATION.md`, `docs/AUDIT.md`, and `docs/LICENSING.md` for training results, implementation details, and licensing information.

- **Language:** Hindi (Devanagari)
- **Voice:** One fixed female voice, `hindi_female`
- **Output:** 24 kHz mono audio
- **Base model:** [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- **Model weights:** [Hugging Face repository](https://huggingface.co/DRG-X/qwen3_TTS_FineTuned-Hindi)

## Quickstart: Inference

Install the inference dependencies and the package:

```bash
pip install -r requirements-inference.txt
pip install -e . --no-deps
```

Generate a Hindi audio sample using the CLI:

```bash
python -m qwen3_tts_hindi \
  "नमस्ते, आप कैसे हैं?" \
  -m DRG-X/qwen3_TTS_FineTuned-Hindi \
  -o out.wav
```

Alternatively, use the Python API:

```python
from qwen3_tts_hindi import HindiTTS

tts = HindiTTS.from_pretrained(
    "DRG-X/qwen3_TTS_FineTuned-Hindi"
)

wav, sr = tts.synthesize(
    "नमस्ते, आप कैसे हैं?",
    seed=0,
)

print(f"Sample rate: {sr} Hz")
```

The exported model has been tested for Hindi speech generation in the author's local environment. The CLI and Python examples document the repository's intended inference interface; reproducibility from a clean installation should be verified in the target environment.

### Important: How Hindi works here

Hindi is not a built-in language ID in the upstream model configuration, so passing `language="Hindi"` may raise a `ValueError`.

This fine-tune uses the `language="auto"` layout with `non_streaming_mode=True`, Devanagari input, and one fixed speaker, `hindi_female`. The `HindiTTS` wrapper is designed to handle these settings.

For lower-level usage, see `examples/raw_qwen_tts_usage.py`.

## What Is in This Repository?

| Path | Purpose |
|---|---|
| `src/qwen3_tts_hindi/` | Inference wrapper and CLI with lazy heavy imports |
| `training/` | Data preparation, LoRA training, and model export scripts |
| `configs/` | Original training configuration and local-path variant |
| `scripts/` | Pipeline runner, adapter conversion, release verification, and training-curve plotting |
| `tests/` | CPU tests, including loss-alignment checks on a small randomly initialized Qwen3-TTS model |
| `docs/` | Training, audit, evaluation, licensing, and release documentation |
| `model_card/README.md` | Model-card source for the Hugging Face release |
| `requirements-inference.txt` | Inference dependencies |
| `requirements-train.txt` | Training dependencies |

## Technical Contributions

1. **Loss-alignment correction:** The library's helper path shifts labels twice for the talker and aligns sub-talker column *j* against codebook *j+2*. The custom `compute_loss` implementation shifts once and evaluates all 15 sub-codebooks. See `tests/test_loss_alignment.py` for the tests.
2. **Hindi without adding a new language ID:** The model was trained using the `auto` and non-streaming layout. Tests check the layout against the intended generation path.
3. **LoRA fine-tuning:** The training configuration targets the talker, text projection, and code predictor, with approximately 36.0 million trainable parameters (1.85% of the model).
4. **Custom-voice export:** The export pipeline prepares a custom-voice-style checkpoint with a speaker embedding. The repository also retains the LoRA adapter artifact and export utilities.
5. **Export validation:** The export workflow includes checks against the base-model snapshot to detect missing assets, including speech-codec weights.

## Results

### Inference smoke test

**Passed in the author's local environment:** The exported model generated multiple Hindi sentences, and the resulting audio was listened to.

This verifies that speech generation works in the tested setup. It is not a formal benchmark of intelligibility, naturalness, pronunciation accuracy, or speaker similarity.

### Teacher-forced validation metrics

The selected checkpoint, at training step 500, achieved the following reported metrics on the 112-utterance validation split:

| Metric | Base model (step 0) | Selected checkpoint (step 500) | Step 750 |
|---|---:|---:|---:|
| Codebook-0 cross-entropy | 1.366 | **0.883** | 0.911 |
| Codebook-0 top-1 accuracy | 0.600 | **0.694** | 0.693 |
| Sub-talker cross-entropy | 5.038 | **4.451** | 4.449 |
| Sub-talker top-1 accuracy | 0.105 | **0.148** | 0.148 |

These metrics measure teacher-forced prediction on validation data. They do not directly measure speech intelligibility or perceived quality.

The validation split was used for checkpoint selection and stopping decisions, so these numbers should not be treated as an unbiased test-set evaluation.

The training logs suggest that overfitting begins after approximately step 250. The training dataset contains approximately 10.2 hours of single-speaker training audio, with 112 validation utterances.

See `docs/EVALUATION.md` for details and caveats.

![Training curves](docs/training_curves.png)

## Reproduce Training

The original training run used two NVIDIA T4 GPUs on Kaggle and took approximately 5.3 hours.

See `docs/TRAINING.md` for setup details.

```bash
pip install -r requirements-train.txt

HTTS_CONFIG=configs/train_hindi_female.local.json \
  bash scripts/run_pipeline.sh
```

The configuration file contains machine-specific paths; adjust them for your environment before running the pipeline.

## Tests

The repository includes CPU tests for loss alignment, configuration, data handling, and export-related behavior.

Run them using:

```bash
pip install torch qwen-tts==0.1.1 safetensors pytest
pip install -e . --no-deps
pytest -q tests
```

The repository's previous CPU test run reported **21 passing tests**. These unit tests do not replace end-to-end audio testing on a clean installation or every supported hardware configuration.

## Limitations

- **One speaker:** The model targets the fixed `hindi_female` voice.
- **Input text:** Devanagari Hindi is the intended input format. Romanized Hindi, English, and code-mixed input have not been systematically evaluated.
- **Long-form text:** Training clips were limited to approximately 15 seconds. Long passages remain untested; split them at sentence boundaries.
- **Text normalization:** Numbers, abbreviations, names, punctuation, and unusual spellings may affect pronunciation.
- **Formal evaluation:** No quantitative character/word error rate, mean opinion score (MOS), or speaker-similarity benchmark has been completed.
- **Runtime performance:** Latency, peak memory usage, and performance across different hardware and precision settings have not been systematically benchmarked.
- **Reproducibility:** The exported model has generated Hindi audio locally, but clean-environment reproducibility of the documented installation and inference commands should be verified.

## Responsible Use

The model is derived from speech data containing a real speaker. Do not use it to impersonate someone, produce deceptive or fraudulent audio, or mislead listeners about whether speech is synthetic. Disclose synthetic audio where listeners could reasonably mistake it for human speech.

## Licensing and Citation

**Code:** See `LICENSE` and `NOTICE` for the repository's code licensing and attribution information.

**Model weights and data:** The base model is [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), and training used [SPRINGLab/IndicTTS-Hindi](https://huggingface.co/datasets/SPRINGLab/IndicTTS-Hindi). Consult `docs/LICENSING.md` and the original dataset terms for the applicable rights and conditions. The code license should not automatically be assumed to apply to every model weight or dataset asset.

To cite this project, see `CITATION.cff`. Please also cite the Qwen3-TTS technical report and the IndicTTS dataset when using this work in research or publications.
