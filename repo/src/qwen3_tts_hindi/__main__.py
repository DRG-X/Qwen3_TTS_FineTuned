"""CLI:  python -m qwen3_tts_hindi "नमस्ते, आप कैसे हैं?" -m <repo-or-dir> -o out.wav"""
import argparse

from .inference import HindiTTS


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthesize Hindi speech with the Qwen3-TTS Hindi fine-tune")
    ap.add_argument("text")
    ap.add_argument("-m", "--model", required=True, help="HF repo id or local directory of the exported model")
    ap.add_argument("-o", "--output", default="out.wav")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    a = ap.parse_args()
    tts = HindiTTS.from_pretrained(a.model, device=a.device, dtype=a.dtype)
    tts.synthesize_to_file(a.text, a.output, seed=a.seed, max_new_tokens=a.max_new_tokens)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
