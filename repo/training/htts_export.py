# htts_export.py — merge the best adapter, save a standalone CustomVoice checkpoint,
# reload it exactly as an end user would, synthesize test sentences, and (optionally)
# score intelligibility with Whisper CER: ground-truth vs base model vs fine-tuned.
import gc
import json
import os
import shutil
import sys
import time
import unicodedata

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htts_common as H

EXTRA_SENTENCES = [
    "नमस्ते, आज का मौसम बहुत सुहावना है।",
    "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।",
    "कृपया अपना टिकट और पहचान पत्र तैयार रखें।",
    "विज्ञान और प्रौद्योगिकी ने हमारे जीवन को पूरी तरह बदल दिया है।",
    "मेरी ट्रेन सुबह सात बजकर पंद्रह मिनट पर छूटेगी।",
]


# ----------------------------------------------------------------------------- helpers
def normalize_hi(s):
    s = unicodedata.normalize("NFC", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return " ".join(s.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    r, h = normalize_hi(ref), normalize_hi(hyp)
    return edit_distance(r, h) / max(1, len(r))


def max_tokens_for(text):
    return int(len(text) * 2.5) + 60   # ~1 frame per Hindi character, 2.5x headroom


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_custom_voice_checkpoint(core, spk, base_dir, out_dir, speaker_name):
    """core: merged Qwen3TTSForConditionalGeneration. Writes a self-contained model dir."""
    from safetensors.torch import save_file
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    # [release-patch P1] original: shutil.copytree(..., ignore=shutil.ignore_patterns("model.safetensors", ...))
    # dropped speech_tokenizer/model.safetensors too (patterns match at every depth). See htts_common.copy_base_assets.
    H.copy_base_assets(base_dir, out_dir)
    with open(os.path.join(base_dir, "config.json"), encoding="utf-8") as f:
        cd = json.load(f)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(H.to_custom_voice_config(cd, speaker_name), f, indent=2, ensure_ascii=False)
    sd = H.custom_voice_state_dict(core, spk)
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    # [release-patch P2] fail loudly if the export is incomplete (this is what the original run lacked)
    problems = H.verify_export_dir(base_dir, out_dir, speaker_name=speaker_name)
    assert not problems, "incomplete export directory:\n  " + "\n  ".join(problems)
    return len(sd)


# ----------------------------------------------------------------------------- main
def main():
    cfg = H.load_cfg()
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem

    out = cfg["output_dir"]
    sdir = os.path.join(out, "samples")
    os.makedirs(sdir, exist_ok=True)
    base_dir = snapshot_download(cfg["base_model"])
    spk = torch.load(os.path.join(out, "speaker_embedding.pt"), map_location="cpu").float()
    adapter_path = os.path.join(out, "adapter_best.pt")
    assert os.path.exists(adapter_path), "no adapter_best.pt — did training run?"
    ck = torch.load(adapter_path, map_location="cpu")
    H.log(f"using adapter from step {ck.get('step')} (val {json.dumps(ck.get('val'))})")

    tests = H.read_jsonl(os.path.join(out, "test_sentences.jsonl"))
    items = [{"id": t["id"], "text": t["text"], "gt_wav": t["wav"]} for t in tests]
    items += [{"id": f"new{i}", "text": s, "gt_wav": None} for i, s in enumerate(EXTRA_SENTENCES)]
    for it in items:
        if it["gt_wav"] and os.path.exists(it["gt_wav"]):
            dst = os.path.join(sdir, f"{it['id']}_groundtruth.wav")
            shutil.copy(it["gt_wav"], dst)
            it["gt_wav"] = dst

    # -------- 1) base model, same x-vector, same (auto, non-streaming) prompt layout
    m = Qwen3TTSModel.from_pretrained(base_dir, device_map="cuda:0", dtype=torch.bfloat16,
                                      attn_implementation=cfg["attn_implementation"])
    prompt = [VoiceClonePromptItem(ref_code=None, ref_spk_embedding=spk.to(torch.bfloat16),
                                   x_vector_only_mode=True, icl_mode=False)]
    if cfg["export_base_samples"]:
        for it in items:
            try:
                torch.manual_seed(0)
                wavs, sr = m.generate_voice_clone(text=it["text"], language="auto", voice_clone_prompt=prompt,
                                                  non_streaming_mode=True, max_new_tokens=max_tokens_for(it["text"]))
                p = os.path.join(sdir, f"{it['id']}_base.wav")
                sf.write(p, wavs[0], sr)
                it["base_wav"] = p
            except Exception as e:  # noqa
                H.log(f"base generation failed for {it['id']}: {e!r}")
        H.log("base-model samples written")

    # -------- 2) inject + load + merge
    core = m.model
    H.inject_lora(core, ck["groups"], dropout=0.0)
    H.load_lora_state_dict(core, ck["lora"])
    n = H.merge_lora(core)
    core.eval()
    H.log(f"merged {n} LoRA layers into base weights")
    model_dir = cfg["export_model_dir"]
    nt = build_custom_voice_checkpoint(core, spk, base_dir, model_dir, cfg["speaker_name"])
    shutil.copy(adapter_path, os.path.join(model_dir, "lora_adapter.pt"))
    shutil.copy(os.path.join(out, "speaker_embedding.pt"), os.path.join(model_dir, "speaker_embedding.pt"))
    H.log(f"saved standalone CustomVoice checkpoint ({nt} tensors) -> {model_dir}")
    del m, core
    free()

    # -------- 3) reload exactly like an end user and synthesize
    ft = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.bfloat16,
                                       attn_implementation=cfg["attn_implementation"])
    H.log(f"reloaded: type={ft.model.tts_model_type} speakers={ft.get_supported_speakers()}")
    t0 = time.time()
    for it in items:
        torch.manual_seed(0)
        wavs, sr = ft.generate_custom_voice(text=it["text"], speaker=cfg["speaker_name"], language="auto",
                                            non_streaming_mode=True, max_new_tokens=max_tokens_for(it["text"]))
        p = os.path.join(sdir, f"{it['id']}_finetuned.wav")
        sf.write(p, wavs[0], sr)
        it["ft_wav"] = p
        it["ft_sec"] = round(len(wavs[0]) / sr, 2)
    H.log(f"fine-tuned samples written ({time.time()-t0:.0f}s)")
    del ft
    free()

    # -------- 4) optional objective intelligibility check
    report = {"adapter_step": ck.get("step"), "adapter_val": ck.get("val"), "items": items}
    if cfg.get("asr_model"):
        try:
            import librosa
            from transformers import pipeline
            asr = pipeline("automatic-speech-recognition", model=cfg["asr_model"],
                           torch_dtype=torch.float16, device=0)

            def transcribe(path):
                y, sr = sf.read(path, dtype="float32")
                if y.ndim > 1:
                    y = y.mean(1)
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                o = asr({"raw": y, "sampling_rate": 16000},
                        generate_kwargs={"language": "hindi", "task": "transcribe"})
                return o["text"].strip()

            agg = {}
            for it in items:
                for kind in ("gt", "base", "ft"):
                    p = it.get(f"{kind}_wav")
                    if p and os.path.exists(p):
                        hyp = transcribe(p)
                        it[f"{kind}_asr"] = hyp
                        it[f"{kind}_cer"] = round(cer(it["text"], hyp), 4)
                        agg.setdefault(kind, []).append(it[f"{kind}_cer"])
            report["mean_cer"] = {k: round(float(np.mean(v)), 4) for k, v in agg.items()}
            report["n_cer"] = {k: len(v) for k, v in agg.items()}
            H.log(f"Whisper CER (lower is better): {report['mean_cer']}  n={report['n_cer']}")
            del asr
            free()
        except Exception as e:  # noqa
            H.log(f"ASR evaluation skipped: {e!r}")
    with open(os.path.join(out, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    H.log("export finished")


if __name__ == "__main__":
    main()
