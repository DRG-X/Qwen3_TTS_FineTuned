# Licensing recommendation and permissions that must be verified

**This is engineering guidance, not legal advice.**

## Recommendation
| Artifact | Recommended licence | Status |
|---|---|---|
| Source code in this repo (`src/`, `scripts/`, `tests/`, docs) | **Apache-2.0** (`LICENSE`, `NOTICE`) | Safe to choose: matches `qwen-tts` and the base model. The `training/` scripts are your own notebook code; they reuse the *structure* of Qwen3-TTS modules and call its library. |
| Model weights (merged model, adapter, speaker embedding) | **Undecided - blocked on the dataset terms.** If the IndicTTS terms permit redistribution of derived models, Apache-2.0 is consistent with the base model; otherwise use a gated/research-only repo with a custom licence, or do not publish weights. | **Do not upload weights until item 2 below is resolved.** |

## Must be verified before publishing weights
1. **Base model** `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and `Qwen/Qwen3-TTS-Tokenizer-12Hz`: secondary sources list Apache-2.0; confirm on the primary model cards and `LICENSE` files on the day of release. Apache-2.0 obliges you to keep the licence, retain notices and state changes (see `NOTICE`).
2. **Dataset** `SPRINGLab/IndicTTS-Hindi`: the Hugging Face card carries no SPDX licence and refers to the original IIT Madras IndicTTS licence terms; a community thread asks what licence applies. Read the IIT Madras "License for use of Indic TTS" terms and decide: may *models trained on* the data be redistributed, commercially, with attribution? If unclear, ask SPRING Lab / IIT Madras in writing.
3. **Voice likeness**: the release encodes one real speaker's voice (a fixed x-vector). Check that the dataset terms/consent cover releasing a model that reproduces that voice, and say so in the model card. Add an acceptable-use note (no impersonation, no deception).
4. **Internship/employer restrictions**: you report that the founder approved. Keep that approval in writing (email) and note any attribution/IP wording they require.
5. **Third-party tools in the pipeline**: Whisper large-v3-turbo (evaluation only; MIT).
6. **Data hygiene**: do not commit dataset audio/text. `test_sentences.jsonl` contains corpus sentences - keep it out of the repo (the 5 sentences in `htts_export.py` are original).
