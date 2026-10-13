"""Minimal reproducible inference example.  UNVERIFIED end to end until the exported model is reloaded on a GPU
(see docs/RELEASE_CHECKLIST.md, step "post-export validation").

    python examples/synthesize.py --model <hf-username>/qwen3-tts-1.7b-hindi-female
"""
import argparse

from qwen3_tts_hindi import HindiTTS

SENTENCES = [
    "नमस्ते, आज का मौसम बहुत सुहावना है।",
    "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "कृपया अपना टिकट और पहचान पत्र तैयार रखें।",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    tts = HindiTTS.from_pretrained(a.model, device=a.device)
    for i, s in enumerate(SENTENCES):
        path = tts.synthesize_to_file(s, f"sample_{i}.wav", seed=0)
        print(path, "<-", s)
