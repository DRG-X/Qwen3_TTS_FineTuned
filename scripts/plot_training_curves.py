"""Re-plot training curves from the REAL train_metrics.jsonl produced by training/htts_train.py."""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "docs/train_metrics.jsonl"
dst = sys.argv[2] if len(sys.argv) > 2 else "docs/training_curves.png"
rows = [json.loads(l) for l in open(src, encoding="utf-8")]
tr = [r for r in rows if r["type"] == "train"]
va = [r for r in rows if r["type"] in ("val", "final")]
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot([r["step"] for r in tr], [r["main"] for r in tr], label="train (10-step avg, dropout on)")
ax[0].plot([r["step"] for r in va], [r["main"] for r in va], "o-", label="val")
ax[0].set_title("codebook-0 CE")
ax[1].plot([r["step"] for r in tr], [r["sub"] for r in tr], label="train")
ax[1].plot([r["step"] for r in va], [r["sub"] for r in va], "o-", label="val")
ax[1].set_title("sub-talker CE (codebooks 1-15)")
ax[2].plot([r["step"] for r in va], [r["main_acc"] for r in va], "o-", label="val codebook-0 top-1")
ax[2].plot([r["step"] for r in va], [r["sub_acc"] for r in va], "o-", label="val sub-talker top-1")
ax[2].set_title("val accuracy")
for a in ax:
    a.set_xlabel("step")
    a.legend()
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(dst, dpi=120)
print("wrote", dst)
