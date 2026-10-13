"""Check an exported model directory against the base-model snapshot it was built from (no weights are loaded).

  python scripts/verify_release_dir.py --base <path to base snapshot> --export <export dir>

Exit code 0 == complete. This is the same check htts_export.py now runs automatically (release-patch P2).
Getting <path to base snapshot>: huggingface_hub.snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base").
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
import htts_common as H

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--export", required=True)
ap.add_argument("--speaker", default="hindi_female")
a = ap.parse_args()
problems = H.verify_export_dir(a.base, a.export, speaker_name=a.speaker)
for p in problems:
    print("PROBLEM:", p)
print("OK" if not problems else f"{len(problems)} problem(s)")
sys.exit(1 if problems else 0)
