"""Convert the pickle-based training artifacts to safetensors (+ JSON metadata) for release.

  python scripts/convert_artifacts_to_safetensors.py --adapter adapter_best.pt --speaker speaker_embedding.pt --out hf_release/adapter

The adapter keeps the notebook's CUSTOM format (keys like talker.model.layers.N.self_attn.q_proj.lora_A); it is NOT a PEFT
adapter. Apply it with training/htts_common.py::inject_lora + load_lora_state_dict (see docs/TRAINING.md).
Round-trip equality is asserted before the script exits.
"""
import argparse
import hashlib
import json
import os

import torch
from safetensors.torch import load_file, save_file


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ck = torch.load(a.adapter, map_location="cpu", weights_only=True)
    lora = {k: v.contiguous() for k, v in ck["lora"].items()}
    p_ad = os.path.join(a.out, "adapter_model.safetensors")
    save_file(lora, p_ad, metadata={"format": "pt", "step": str(ck["step"])})
    back = load_file(p_ad)
    assert back.keys() == lora.keys() and all(torch.equal(back[k], lora[k]) for k in lora), "round-trip mismatch"

    cfg = {
        "format": "qwen3-tts-hindi custom LoRA (not PEFT)",
        "step": int(ck["step"]),
        "val_at_checkpoint": ck["val"],
        "best_val_total": ck["best_val"],
        "groups_rank_alpha": {k: list(v) for k, v in ck["groups"].items()},
        "num_tensors": len(lora),
        "num_parameters": sum(v.numel() for v in lora.values()),
        "dtype": "float32",
        "sha256": {"adapter_model.safetensors": sha256(p_ad)},
    }

    spk = torch.load(a.speaker, map_location="cpu", weights_only=True).float().contiguous()
    p_sp = os.path.join(a.out, "speaker_embedding.safetensors")
    save_file({"speaker_embedding": spk}, p_sp, metadata={"format": "pt"})
    assert torch.equal(load_file(p_sp)["speaker_embedding"], spk)
    cfg["speaker_embedding"] = {"shape": list(spk.shape), "norm": float(spk.norm()), "sha256": sha256(p_sp)}

    with open(os.path.join(a.out, "adapter_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
