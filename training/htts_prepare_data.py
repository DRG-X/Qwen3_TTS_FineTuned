# htts_prepare_data.py — one-GPU data preparation (resumable)
#  1) female Hindi utterances -> 24 kHz mono, trimmed, peak-normalised wavs
#  2) 12 Hz codec codes (fp32 tokenizer, length-sorted batches)
#  3) text token ids (same template as inference) + structural assertions
#  4) ONE fixed speaker x-vector (mean of many clips) -> used for training AND export
#  5) train / val / test split
import io
import os
import random
import sys
import time

import numpy as np
import soundfile as sf
import librosa
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htts_common as H

cfg = H.load_cfg()
SR = 24000
D = cfg["data_dir"]
WAV_DIR = os.path.join(D, "wavs")
os.makedirs(WAV_DIR, exist_ok=True)
os.makedirs(cfg["output_dir"], exist_ok=True)
RAW = os.path.join(D, "raw.jsonl")
CODED = os.path.join(D, "coded.jsonl")
rng = random.Random(cfg["seed"])


def clean_text(t):
    return " ".join(str(t).replace("\u200b", "").split()).strip()


def process_wave(y, sr):
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    yt, _ = librosa.effects.trim(y, top_db=cfg["trim_top_db"], frame_length=1024, hop_length=256)
    pad = np.zeros(int(0.12 * SR), dtype=np.float32)
    y = np.concatenate([pad, yt, pad])
    peak = float(np.abs(y).max() + 1e-9)
    return (y * (0.9 / peak)).astype(np.float32)


# ----------------------------------------------------------------------------- step 1
def materialize_hf():
    from datasets import load_dataset, Audio
    ds = load_dataset(cfg["hf_dataset"], split="train")
    names = ds.features["gender"].names
    fem = names.index("female")
    ds = ds.filter(lambda g: g == fem, input_columns=["gender"], num_proc=2)
    ds = ds.cast_column("audio", Audio(decode=False))  # decode ourselves (no torchcodec dependency)
    H.log(f"female utterances in {cfg['hf_dataset']}: {len(ds)}")
    rows = []
    for i in range(len(ds)):
        ex = ds[i]
        text = clean_text(ex["text"])
        if not text:
            continue
        a = ex["audio"]
        try:
            if a.get("bytes"):
                y, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32", always_2d=False)
            else:
                y, sr = sf.read(a["path"], dtype="float32", always_2d=False)
        except Exception as e:  # noqa
            print("skip unreadable", i, e)
            continue
        rows.append((f"f{i:06d}", y, sr, text))
        if len(rows) >= 256:
            yield rows
            rows = []
    if rows:
        yield rows


def materialize_local():
    """Generic local layout: <dir>/*.wav + transcripts.csv (filename,text) or same-stem .txt"""
    import csv
    root = cfg["local_dataset_dir"]
    tmap = {}
    csvp = os.path.join(root, "transcripts.csv")
    if os.path.exists(csvp):
        with open(csvp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tmap[r["filename"]] = r["text"]
    wavs = sorted(f for f in os.listdir(root) if f.lower().endswith(".wav"))
    rows = []
    for i, fn in enumerate(wavs):
        text = tmap.get(fn)
        if text is None:
            tp = os.path.join(root, os.path.splitext(fn)[0] + ".txt")
            text = open(tp, encoding="utf-8").read() if os.path.exists(tp) else ""
        text = clean_text(text)
        if not text:
            continue
        y, sr = sf.read(os.path.join(root, fn), dtype="float32", always_2d=False)
        rows.append((f"l{i:06d}", y, sr, text))
        if len(rows) >= 256:
            yield rows
            rows = []
    if rows:
        yield rows


def step1():
    if os.path.exists(RAW):
        H.log(f"step1 skipped ({RAW} exists)")
        return H.read_jsonl(RAW)
    src = materialize_hf() if cfg["dataset_source"] == "hf" else materialize_local()
    out, dropped = [], 0
    t0 = time.time()
    for chunk in src:
        for uid, y, sr, text in chunk:
            y = process_wave(y, sr)
            dur = len(y) / SR
            if not (cfg["min_audio_sec"] <= dur <= cfg["max_audio_sec"]):
                dropped += 1
                continue
            path = os.path.join(WAV_DIR, uid + ".wav")
            sf.write(path, y, SR, subtype="PCM_16")
            out.append({"id": uid, "wav": path, "text": text, "duration": round(dur, 3)})
        H.log(f"  materialized {len(out)} (dropped {dropped} by duration) in {time.time()-t0:.0f}s")
    if cfg.get("max_utterances"):
        rng.shuffle(out)
        out = out[: cfg["max_utterances"]]
    H.write_jsonl(RAW, out)
    hours = sum(r["duration"] for r in out) / 3600
    H.log(f"step1 done: {len(out)} utterances, {hours:.2f} h of audio")
    return out


# ----------------------------------------------------------------------------- step 2
def step2(rows):
    from qwen_tts import Qwen3TTSTokenizer
    done = {r["id"]: r for r in H.read_jsonl(CODED)} if os.path.exists(CODED) else {}
    todo = sorted([r for r in rows if r["id"] not in done], key=lambda r: r["duration"])
    if not todo:
        H.log("step2 skipped (all coded)")
        return [done[r["id"]] for r in rows if r["id"] in done]
    tok = Qwen3TTSTokenizer.from_pretrained(cfg["tokenizer_model"], device_map="cuda:0")  # default fp32
    f = open(CODED, "a", encoding="utf-8")
    bs_sec = cfg["codec_batch_seconds"]
    i, nb, t0 = 0, 0, time.time()
    import json
    while i < len(todo):
        batch, tot = [], 0.0
        while i < len(todo) and (not batch or tot + todo[i]["duration"] <= bs_sec):
            batch.append(todo[i]); tot += todo[i]["duration"]; i += 1
        with torch.inference_mode():
            enc = tok.encode([r["wav"] for r in batch])
        for r, codes in zip(batch, enc.audio_codes):
            codes = codes.cpu()
            assert codes.dim() == 2 and codes.shape[1] == 16, codes.shape
            assert int(codes.max()) < 2048 and int(codes.min()) >= 0
            r = dict(r, audio_codes=codes.tolist())
            done[r["id"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        nb += 1
        if nb % 25 == 0 or i == len(todo):
            H.log(f"  coded {i}/{len(todo)}  ({time.time()-t0:.0f}s)")
    f.close()
    del tok
    torch.cuda.empty_cache()
    fr = [len(done[r["id"]]["audio_codes"]) / done[r["id"]]["duration"] for r in rows[:50] if r["id"] in done]
    H.log(f"step2 done; codec frame-rate check ≈ {np.mean(fr):.2f} frames/s (expect ~12.5)")
    return [done[r["id"]] for r in rows if r["id"] in done]


# ----------------------------------------------------------------------------- step 3/4
def step34(rows):
    from qwen_tts import Qwen3TTSModel
    from huggingface_hub import snapshot_download
    model_dir = snapshot_download(cfg["base_model"])
    m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cuda:0", dtype=torch.bfloat16,
                                      attn_implementation="sdpa")
    hf_cfg = m.model.config
    assert hf_cfg.tts_model_type == "base", hf_cfg.tts_model_type
    assert hf_cfg.talker_config.num_code_groups == 16
    H.log(f"talker hidden={hf_cfg.talker_config.hidden_size} text_hidden={hf_cfg.talker_config.text_hidden_size} "
          f"cp_hidden={hf_cfg.talker_config.code_predictor_config.hidden_size} "
          f"supported languages={m.get_supported_languages()}")

    n_tok = []
    for r in rows:
        r["text_ids"] = H.tokenize_for_training(m.processor, r["text"], hf_cfg)
        n_tok.append(len(r["text_ids"]) - 3)
    chars = np.mean([len(r["text"]) for r in rows])
    H.log(f"text tokens/utterance: mean {np.mean(n_tok):.1f}, max {max(n_tok)}; chars/utt {chars:.1f}")
    rows = [r for r in rows if H.seq_len_of(r) <= cfg["max_seq_len"]]

    # speaker x-vector: average over many clean 3-10 s clips (fp32 encoder)
    cands = [r for r in rows if 3.0 <= r["duration"] <= 10.0]
    rng.shuffle(cands)
    cands = cands[: cfg["num_speaker_clips"]]
    enc = m.model.speaker_encoder.float()
    embs = []
    for r in cands:
        y, _ = sf.read(r["wav"], dtype="float32")
        mel = H_mel(y)
        with torch.inference_mode():
            embs.append(enc(mel.to("cuda:0")).float()[0].cpu())
    E = torch.stack(embs)
    mean = E.mean(0)
    mean = mean * (E.norm(dim=-1).mean() / mean.norm())
    cos = torch.nn.functional.cosine_similarity(E, mean.unsqueeze(0)).mean().item()
    torch.save(mean, os.path.join(cfg["output_dir"], "speaker_embedding.pt"))
    H.log(f"speaker x-vector from {len(E)} clips, dim={mean.numel()}, mean cos-sim to centroid={cos:.3f}")
    # keep one reference clip for later zero-shot/voice-clone comparisons
    import shutil
    shutil.copy(cands[0]["wav"], os.path.join(cfg["output_dir"], "reference_clip.wav"))
    with open(os.path.join(cfg["output_dir"], "reference_clip.txt"), "w", encoding="utf-8") as f:
        f.write(cands[0]["text"])
    del m, enc
    torch.cuda.empty_cache()
    return rows


def H_mel(y):
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
    return mel_spectrogram(torch.from_numpy(y).unsqueeze(0), n_fft=1024, num_mels=128, sampling_rate=SR,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000).transpose(1, 2)


# ----------------------------------------------------------------------------- step 5
def step5(rows):
    rows = sorted(rows, key=lambda r: r["id"])
    rng.shuffle(rows)
    n_val = min(cfg["val_max"], max(cfg["val_min"], int(len(rows) * cfg["val_ratio"])))
    val, train = rows[:n_val], rows[n_val:]
    test = [r for r in val if 2.5 <= r["duration"] <= 9.0][: cfg["num_test_sentences"]]
    H.write_jsonl(cfg["train_jsonl"], train)
    H.write_jsonl(cfg["val_jsonl"], val)
    H.write_jsonl(os.path.join(cfg["output_dir"], "test_sentences.jsonl"),
                  [{"id": r["id"], "text": r["text"], "wav": r["wav"]} for r in test])
    th = sum(r["duration"] for r in train) / 3600
    H.log(f"split: train {len(train)} ({th:.2f} h) | val {len(val)} | listening-test {len(test)}")


if __name__ == "__main__":
    if os.path.exists(cfg["train_jsonl"]) and os.path.exists(cfg["val_jsonl"]):
        H.log("prepared data already present — nothing to do")
        sys.exit(0)
    t0 = time.time()
    rows = step1()
    rows = step2(rows)
    rows = step34(rows)
    step5(rows)
    H.log(f"data prep finished in {(time.time()-t0)/60:.1f} min")
