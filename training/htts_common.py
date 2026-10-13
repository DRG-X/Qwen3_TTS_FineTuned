# htts_common.py — shared code for Qwen3-TTS-12Hz-1.7B-Base Hindi fine-tuning
# ---------------------------------------------------------------------------
# Everything that must be IDENTICAL between data prep, training and export
# lives here: config loading, the sequence layout (collate), LoRA, and the loss.
#
# LOSS ALIGNMENT (the core fix). Verified against qwen-tts==0.1.1 +
# transformers==4.57.3:
#   * Talker.forward(labels=...) and CodePredictor.forward_finetune(labels=...)
#     both call `self.loss_function`, which resolves to HF `ForCausalLMLoss`.
#     That function ALREADY shifts labels by one position.
#   * The official sft_12hz.py (and most community scripts) ALSO shift manually
#     (inputs[:, :-1], labels[:, 1:])  ->  double shift: every position is
#     trained to predict the token TWO steps ahead.
#   * forward_sub_talker_finetune() builds logits that are ALREADY aligned with
#     codec_ids[:, 1:] (logit j = lm_head[j] at position j+1 -> codebook j+1),
#     then hands them to ForCausalLMLoss, which shifts again -> lm_head[j] is
#     trained to emit codebook j+2. No script can fix this from outside except
#     by NOT using that returned loss. We compute both cross-entropies here.
# ---------------------------------------------------------------------------
import json
import math
import os
import random
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

IGNORE_INDEX = -100
CUSTOM_SPEAKER_CODEC_ID = 3000  # same slot the official exporter uses


# ----------------------------------------------------------------------------
# Config / IO helpers
# ----------------------------------------------------------------------------
def load_cfg(path=None):
    path = path or os.environ.get("HTTS_CONFIG", "/kaggle/working/htts/config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def log(msg, rank=0):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def seconds_left(cfg):
    """Seconds left before the training deadline (measured from notebook start)."""
    start = cfg.get("notebook_start_ts") or time.time()
    deadline = start + float(cfg["max_total_hours"]) * 3600.0 - float(cfg["export_reserve_min"]) * 60.0
    return deadline - time.time()


# ----------------------------------------------------------------------------
# Text tokenisation (identical to Qwen3TTSModel._build_assistant_text + slicing)
# ----------------------------------------------------------------------------
def build_assistant_text(text):
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def tokenize_for_training(processor, text, hf_cfg):
    """Returns python list: [<|im_start|>, assistant, \\n, text tokens...].
    The 5-token suffix (<|im_end|> \\n <|im_start|> assistant \\n) is removed,
    exactly as inference does with input_id[:, 3:-5]."""
    ids = processor(text=build_assistant_text(text), return_tensors="pt", padding=True)["input_ids"]
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids[0].tolist()
    # Structural sanity checks — fail loudly instead of training on a wrong layout.
    assert ids[0] == hf_cfg.im_start_token_id, f"expected <|im_start|> first, got {ids[:4]}"
    assert ids[-5] == hf_cfg.im_end_token_id, f"expected <|im_end|> at -5, got {ids[-6:]}"
    assert ids[-3] == hf_cfg.im_start_token_id, f"expected <|im_start|> at -3, got {ids[-6:]}"
    return ids[:-5]


# ----------------------------------------------------------------------------
# Dataset + collate (same token layout as official dataset.py = the
# NON-STREAMING, language="auto" prompt that generate() builds)
# ----------------------------------------------------------------------------
class HindiTTSDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "text_ids": torch.tensor(r["text_ids"], dtype=torch.long),
            "audio_codes": torch.tensor(r["audio_codes"], dtype=torch.long),  # (N, 16)
        }


def seq_len_of(row):
    return len(row["text_ids"]) + len(row["audio_codes"]) + 8


class TTSCollator:
    """Layout per sample (L = len(text_ids) incl. 3 role tokens, N = frames):
       pos 0..2            text: role tokens            codec: -
       pos 3..6            text: tts_pad                codec: nothink, think_bos, think_eos, [SPEAKER]
       pos 7               text: tts_bos                codec: codec_pad
       pos 8..8+L-4        text: Hindi tokens           codec: codec_pad
       pos 8+L-3           text: tts_eos                codec: codec_pad
       pos 8+L-2           text: tts_pad                codec: codec_bos
       pos s=8+L-1..s+N-1  text: tts_pad                codec: frame k (all 16 codebooks summed)
       pos s+N             text: tts_pad                codec: codec_eos
       labels[p] == codebook-0 token present at input position p (HF convention);
       the loss pairs logits[p-1] with labels[p] exactly once."""

    def __init__(self, hf_cfg):
        self.c = hf_cfg
        self.t = hf_cfg.talker_config
        self.num_groups = hf_cfg.talker_config.num_code_groups

    def __call__(self, batch):
        c, t = self.c, self.t
        B = len(batch)
        T = max(b["text_ids"].numel() + b["audio_codes"].shape[0] for b in batch) + 8
        text_in = torch.zeros((B, T), dtype=torch.long)
        codec_in = torch.zeros((B, T), dtype=torch.long)
        codec_ids = torch.zeros((B, T, self.num_groups), dtype=torch.long)
        text_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_emb_mask = torch.zeros((B, T), dtype=torch.bool)
        codec_mask = torch.zeros((B, T), dtype=torch.bool)
        attn = torch.zeros((B, T), dtype=torch.long)
        labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)

        for i, b in enumerate(batch):
            ti = b["text_ids"].view(-1)
            ac = b["audio_codes"]
            L, N = ti.numel(), ac.shape[0]
            s = 8 + L - 1
            end = 8 + L + N  # exclusive

            text_in[i, :3] = ti[:3]
            text_in[i, 3:7] = c.tts_pad_token_id
            text_in[i, 7] = c.tts_bos_token_id
            text_in[i, 8:8 + L - 3] = ti[3:]
            text_in[i, 8 + L - 3] = c.tts_eos_token_id
            text_in[i, 8 + L - 2:end] = c.tts_pad_token_id
            text_mask[i, :end] = True

            codec_in[i, 3:8] = torch.tensor(
                [t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id, 0, t.codec_pad_id]
            )
            codec_in[i, 8:8 + L - 2] = t.codec_pad_id
            codec_in[i, 8 + L - 2] = t.codec_bos_id
            codec_in[i, s:s + N] = ac[:, 0]
            codec_in[i, s + N] = t.codec_eos_token_id
            codec_emb_mask[i, 3:end] = True
            codec_emb_mask[i, 6] = False  # speaker slot is filled with the x-vector

            codec_ids[i, s:s + N] = ac
            codec_mask[i, s:s + N] = True
            labels[i, s:s + N] = ac[:, 0]
            labels[i, s + N] = t.codec_eos_token_id
            attn[i, :end] = 1

        return {
            "text_in": text_in, "codec_in": codec_in, "codec_ids": codec_ids,
            "text_mask": text_mask, "codec_emb_mask": codec_emb_mask,
            "codec_mask": codec_mask, "attn": attn, "labels": labels,
        }


class LengthGroupedBatchSampler:
    """Deterministic (seed+epoch) length-bucketed batches, sharded so every rank
    gets the SAME number of batches (required by DDP)."""

    def __init__(self, lengths, batch_size, world_size=1, rank=0, seed=0, shuffle=True):
        self.lengths, self.bs = lengths, batch_size
        self.world, self.rank, self.seed, self.shuffle = world_size, rank, seed, shuffle
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _all_batches(self):
        n = len(self.lengths)
        idx = list(range(n))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
            mega = self.bs * 50
            chunks = [idx[s:s + mega] for s in range(0, n, mega)]
        else:
            chunks = [idx]
        batches = []
        for ch in chunks:
            ch = sorted(ch, key=lambda i: self.lengths[i], reverse=True)
            batches += [ch[j:j + self.bs] for j in range(0, len(ch), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        usable = (len(batches) // self.world) * self.world
        if usable == 0:
            usable = len(batches)  # tiny eval sets: ranks may differ; eval never syncs per-batch
        return batches[:usable]

    def __iter__(self):
        yield from self._all_batches()[self.rank::self.world]

    def __len__(self):
        return len(self._all_batches()[self.rank::self.world])


def move_batch(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ----------------------------------------------------------------------------
# Minimal LoRA (no PEFT wrapper -> no attribute-forwarding surprises)
# ----------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # some HF code peeks at .weight / .bias of projection layers
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        out = self.base(x)
        d = self.drop(x).to(self.lora_A.dtype)
        d = F.linear(F.linear(d, self.lora_A), self.lora_B) * self.scaling
        return out + d.to(out.dtype)

    @torch.no_grad()
    def merged_base(self):
        w = self.base.weight
        delta = (self.lora_B @ self.lora_A) * self.scaling
        w.add_(delta.to(device=w.device, dtype=w.dtype))
        return self.base


_PROJ = r"(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
LORA_PATTERNS = {
    "talker": rf"^talker\.model\.layers\.\d+\.{_PROJ}$",
    "text_projection": r"^talker\.text_projection\.linear_fc[12]$",
    "code_predictor": rf"^talker\.code_predictor\.model\.layers\.\d+\.{_PROJ}$",
}


def lora_groups_from_cfg(cfg):
    g = {"talker": (cfg["lora_r"], cfg["lora_alpha"])}
    if cfg.get("lora_text_projection", True):
        g["text_projection"] = (cfg["lora_r"], cfg["lora_alpha"])
    if cfg.get("lora_code_predictor", True):
        g["code_predictor"] = (cfg["cp_lora_r"], cfg["cp_lora_alpha"])
    return g


def inject_lora(core, groups, dropout):
    """groups: {group_name: (r, alpha)}. Freezes everything, then wraps matched Linear layers."""
    for p in core.parameters():
        p.requires_grad_(False)
    compiled = {g: re.compile(LORA_PATTERNS[g]) for g in groups}
    targets = []
    for name, mod in core.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        for g, rx in compiled.items():
            if rx.match(name):
                targets.append((name, g))
                break
    counts = {g: 0 for g in groups}
    for name, g in targets:
        parent_name, child = name.rsplit(".", 1)
        parent = core.get_submodule(parent_name)
        r, alpha = groups[g]
        setattr(parent, child, LoRALinear(getattr(parent, child), r, alpha, dropout))
        counts[g] += 1
    for g, n in counts.items():
        assert n > 0, f"LoRA group '{g}' matched no layers — module names changed?"
    return counts


def lora_state_dict(core):
    return {n: p.detach().float().cpu().clone() for n, p in core.named_parameters() if ".lora_" in n}


def load_lora_state_dict(core, sd):
    own = dict(core.named_parameters())
    missing = [k for k in own if ".lora_" in k and k not in sd]
    unexpected = [k for k in sd if k not in own]
    assert not missing and not unexpected, f"LoRA mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v.to(own[k].device, own[k].dtype))


def merge_lora(core):
    names = [n for n, m in core.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = core.get_submodule(parent_name)
        setattr(parent, child, getattr(parent, child).merged_base())
    return len(names)


# ----------------------------------------------------------------------------
# Input embeddings (shared by corrected and legacy losses)
# ----------------------------------------------------------------------------
def build_talker_inputs(core, batch, spk_emb, use_text_projection=True):
    talker = core.talker
    txt = talker.model.text_embedding(batch["text_in"])
    if use_text_projection:
        txt = talker.text_projection(txt)
    txt = txt * batch["text_mask"].unsqueeze(-1).to(txt.dtype)
    cod = talker.model.codec_embedding(batch["codec_in"])
    cod = cod * batch["codec_emb_mask"].unsqueeze(-1).to(cod.dtype)
    spk = spk_emb.to(device=cod.device, dtype=cod.dtype)
    if spk.dim() == 1:
        spk = spk.unsqueeze(0).expand(cod.shape[0], -1)
    cod = torch.cat([cod[:, :6], spk.unsqueeze(1), cod[:, 7:]], dim=1)
    x = txt.to(cod.dtype) + cod
    cp_emb = talker.code_predictor.get_input_embeddings()
    fmask = batch["codec_mask"].unsqueeze(-1).to(x.dtype)
    for k in range(1, talker.config.num_code_groups):
        x = x + cp_emb[k - 1](batch["codec_ids"][:, :, k]) * fmask
    return x


def _sub_talker_inputs(talker, frame_hidden, frame_codes):
    cp_emb = talker.code_predictor.get_input_embeddings()
    G = talker.config.num_code_groups
    parts = [frame_hidden.unsqueeze(1), talker.get_input_embeddings()(frame_codes[:, :1])]
    for j in range(1, G - 1):
        parts.append(cp_emb[j - 1](frame_codes[:, j:j + 1]))
    return torch.cat(parts, dim=1)  # (F, G, H_talker)


# ----------------------------------------------------------------------------
# CORRECT loss
# ----------------------------------------------------------------------------
def compute_loss(core, batch, spk_emb, sub_weight):
    talker = core.talker
    x = build_talker_inputs(core, batch, spk_emb, use_text_projection=True)

    # Feed T-1 positions, NO labels -> the model computes no loss internally.
    out = talker(inputs_embeds=x[:, :-1], attention_mask=batch["attn"][:, :-1], output_hidden_states=True)
    logits = out.logits.float()                      # logits[p] saw inputs 0..p
    target = batch["labels"][:, 1:]                  # target[p] = codebook-0 at p+1   (single shift)
    V = logits.shape[-1]
    main_loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), ignore_index=IGNORE_INDEX)
    with torch.no_grad():
        valid = target != IGNORE_INDEX
        main_acc = (logits.argmax(-1)[valid] == target[valid]).float().mean()
        n_main = valid.sum()

    # Sub-talker: hidden[p-1] (post-norm, the state that PREDICTED frame p's codebook-0)
    # is paired with frame p's 16 codes — identical to inference (past_hidden + code0).
    hidden = out.hidden_states[0][-1]                # (B, T-1, H), post final norm
    frame_hidden = hidden[batch["codec_mask"][:, 1:]]
    frame_codes = batch["codec_ids"][batch["codec_mask"]]  # (F, 16)
    sub_in = _sub_talker_inputs(talker, frame_hidden, frame_codes)
    sub_out = talker.code_predictor.forward_finetune(inputs_embeds=sub_in)  # labels=None
    sub_logits = sub_out.logits.float()              # (F, 15, V2); column j predicts codebook j+1
    sub_target = frame_codes[:, 1:]
    V2 = sub_logits.shape[-1]
    sub_loss = F.cross_entropy(sub_logits.reshape(-1, V2), sub_target.reshape(-1))
    with torch.no_grad():
        sub_acc = (sub_logits.argmax(-1) == sub_target).float().mean()

    total = main_loss + sub_weight * sub_loss
    stats = {
        "main": main_loss.detach(), "sub": sub_loss.detach(),
        "main_acc": main_acc, "sub_acc": sub_acc, "n": n_main.float(),
    }
    return total, stats


# ----------------------------------------------------------------------------
# LEGACY loss (what the previous notebook computed) — diagnostics only
# ----------------------------------------------------------------------------
@torch.no_grad()
def compute_legacy_loss(core, batch, spk_emb):
    talker = core.talker
    x = build_talker_inputs(core, batch, spk_emb, use_text_projection=True)
    out = talker(
        inputs_embeds=x[:, :-1], attention_mask=batch["attn"][:, :-1],
        labels=batch["labels"][:, 1:], output_hidden_states=True,   # double shift
    )
    hidden = out.hidden_states[0][-1]
    frame_hidden = hidden[batch["codec_mask"][:, 1:]]
    frame_codes = batch["codec_ids"][batch["codec_mask"]]
    _, sub_loss = talker.forward_sub_talker_finetune(frame_codes, frame_hidden)  # internal double shift
    return out.loss.float(), sub_loss.float()


# ----------------------------------------------------------------------------
# DDP-friendly wrapper (DDP needs gradients to flow through module.forward)
# ----------------------------------------------------------------------------
class TrainWrapper(nn.Module):
    def __init__(self, core, spk_emb, sub_weight):
        super().__init__()
        self.core = core
        self.register_buffer("spk_emb", spk_emb.clone(), persistent=False)
        self.sub_weight = sub_weight

    def forward(self, batch):
        return compute_loss(self.core, batch, self.spk_emb, self.sub_weight)


# ----------------------------------------------------------------------------
# Custom-voice export (official format: speaker x-vector baked into codec slot 3000)
# ----------------------------------------------------------------------------
def to_custom_voice_config(config_dict, speaker_name):
    cd = json.loads(json.dumps(config_dict))
    cd["tts_model_type"] = "custom_voice"
    tc = cd.setdefault("talker_config", {})
    tc["spk_id"] = {speaker_name.lower(): CUSTOM_SPEAKER_CODEC_ID}
    tc["spk_is_dialect"] = {speaker_name.lower(): False}
    return cd


def custom_voice_state_dict(core, spk_emb):
    sd = {k: v.detach().to("cpu").contiguous() for k, v in core.state_dict().items()
          if not k.startswith("speaker_encoder")}
    key = "talker.model.codec_embedding.weight"
    w = sd[key].clone()
    w[CUSTOM_SPEAKER_CODEC_ID] = spk_emb.to(w.dtype).cpu()
    sd[key] = w
    return sd


# ----------------------------------------------------------------------------
# [release-patch P1/P2] ADDED for the open-source release; NOT in the original notebook.
#
# Why: the original export copied the base model with
#     shutil.copytree(..., ignore=shutil.ignore_patterns("model.safetensors", "model*.safetensors", ...))
# `ignore_patterns` matches at EVERY directory depth, so it also dropped
# speech_tokenizer/model.safetensors (the codec weights). The exported folder then
# failed to reload with "no file named ... model.safetensors ... speech_tokenizer".
# Fix: skip the weight files only at the TOP level of the base snapshot, and verify
# the result against the actual base snapshot (no hard-coded file list).
# ----------------------------------------------------------------------------
TOP_LEVEL_WEIGHT_PATTERNS = ("model.safetensors", "model*.safetensors", "model.safetensors.index.json")


def _rel_files(root):
    out = set()
    for dp, _, fs in os.walk(root):
        for f in fs:
            out.add(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
    return out


def _is_skipped_from_base(rel):
    import fnmatch
    name = rel.rsplit("/", 1)[-1]
    if fnmatch.fnmatch(name, ".git*"):
        return True
    return "/" not in rel and any(fnmatch.fnmatch(rel, p) for p in TOP_LEVEL_WEIGHT_PATTERNS)


def copy_base_assets(base_dir, out_dir):
    """Copy every non-weight asset (config, tokenizer, processor, speech_tokenizer/ incl. its weights)."""
    import fnmatch
    import shutil
    base_abs = os.path.abspath(base_dir)

    def ignore(directory, names):
        skipped = {n for n in names if fnmatch.fnmatch(n, ".git*")}
        if os.path.abspath(directory) == base_abs:
            skipped |= {n for n in names if any(fnmatch.fnmatch(n, p) for p in TOP_LEVEL_WEIGHT_PATTERNS)}
        return skipped

    shutil.copytree(base_dir, out_dir, symlinks=False, ignore=ignore)


def verify_export_dir(base_dir, out_dir, speaker_name=None, check_tensor_keys=True):
    """Return a list of problems (empty list == export directory is complete)."""
    problems = []
    expected = {r for r in _rel_files(base_dir) if not _is_skipped_from_base(r)}
    have = _rel_files(out_dir)
    for rel in sorted(expected - have):
        problems.append(f"missing from export (present in base snapshot): {rel}")
    if "model.safetensors" not in have:
        problems.append("missing top-level model.safetensors")
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            c = json.load(f)
        if c.get("tts_model_type") != "custom_voice":
            problems.append(f"config.tts_model_type is {c.get('tts_model_type')!r}, expected 'custom_voice'")
        if speaker_name is not None:
            spk = c.get("talker_config", {}).get("spk_id", {})
            if spk.get(speaker_name.lower()) != CUSTOM_SPEAKER_CODEC_ID:
                problems.append(f"config.talker_config.spk_id lacks {speaker_name.lower()!r}: {CUSTOM_SPEAKER_CODEC_ID}")
    if check_tensor_keys and "model.safetensors" in have:
        try:
            from safetensors import safe_open
            with safe_open(os.path.join(out_dir, "model.safetensors"), framework="pt") as f:
                keys = set(f.keys())
            if any(k.startswith("speaker_encoder") for k in keys):
                problems.append("model.safetensors still contains speaker_encoder.* tensors")
            if "talker.model.codec_embedding.weight" not in keys:
                problems.append("model.safetensors lacks talker.model.codec_embedding.weight")
            if any(".lora_" in k or k.endswith(".base.weight") for k in keys):
                problems.append("model.safetensors contains un-merged LoRA wrapper keys")
        except Exception as e:  # noqa
            problems.append(f"could not read model.safetensors keys: {e!r}")
    return problems
