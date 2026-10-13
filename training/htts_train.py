# htts_train.py — LoRA fine-tuning of Qwen3-TTS-12Hz-1.7B-Base on Hindi
# Launch:  torchrun --nproc_per_node=<gpus> htts_train.py      (or plain `python htts_train.py`)
import contextlib
import datetime
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import htts_common as H


def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device, backend = torch.device("cuda", local), "nccl"
    else:
        device, backend = torch.device("cpu"), "gloo"
    if world > 1:
        dist.init_process_group(backend, timeout=datetime.timedelta(minutes=60))
    return world, rank, local, device


def all_reduce(t, op=None):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=op or dist.ReduceOp.SUM)
    return t


def load_core(cfg, device):
    """Returns (core nn.Module, hf_config). Overridden in CPU tests."""
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel
    model_dir = snapshot_download(cfg["base_model"])
    m = Qwen3TTSModel.from_pretrained(model_dir, device_map=str(device), dtype=torch.bfloat16,
                                      attn_implementation=cfg["attn_implementation"])
    core = m.model
    core.speech_tokenizer = None      # codec decoder not needed for training
    core.speaker_encoder = None       # fixed x-vector is precomputed
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return core, core.config


def save_adapter(core, path, meta):
    tmp = path + ".tmp"
    torch.save({"lora": H.lora_state_dict(core), **meta}, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(wrapper, loader, device):
    core = wrapper.core
    was_training = core.training
    core.eval()
    acc = torch.zeros(5, device=device, dtype=torch.float64)
    for batch in loader:
        batch = H.move_batch(batch, device)
        _, st = wrapper(batch)
        n = st["n"].double()
        acc += torch.stack([st["main"].double() * n, st["sub"].double() * n,
                            st["main_acc"].double() * n, st["sub_acc"].double() * n, n])
    all_reduce(acc)
    if was_training:
        core.train()
    n = acc[4].clamp_min(1)
    main, sub = (acc[0] / n).item(), (acc[1] / n).item()
    return {"main": main, "sub": sub, "total": main + wrapper.sub_weight * sub,
            "main_acc": (acc[2] / n).item(), "sub_acc": (acc[3] / n).item(), "tokens": int(acc[4].item())}


@torch.no_grad()
def alignment_diagnostic(wrapper, loader, device, rank, n_batches):
    """Prints corrected vs legacy (double-shifted) loss on the untouched base model."""
    core = wrapper.core
    core.eval()
    ok = torch.zeros(6, device=device, dtype=torch.float64)
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batch = H.move_batch(batch, device)
        _, st = wrapper(batch)
        lm, ls = H.compute_legacy_loss(core, batch, wrapper.spk_emb)
        ok += torch.tensor([st["main"].item(), st["sub"].item(), lm.item(), ls.item(), st["main_acc"].item(), 1.0],
                           device=device, dtype=torch.float64)
    all_reduce(ok)
    k = ok[5].clamp_min(1)
    v = (ok / k).tolist()
    core.train()
    H.log("=" * 78, rank)
    H.log("ALIGNMENT DIAGNOSTIC on the base model (before any update):", rank)
    H.log(f"  corrected  : codebook-0 CE = {v[0]:.3f}   sub-talker CE = {v[1]:.3f}   c0 top-1 acc = {v[4]:.3f}", rank)
    H.log(f"  legacy     : codebook-0 CE = {v[2]:.3f}   sub-talker CE = {v[3]:.3f}   (previous notebook)", rank)
    H.log(f"  uniform-guess reference: ln(3072) = {math.log(3072):.2f}, ln(2048) = {math.log(2048):.2f}", rank)
    H.log("  A pretrained model scores far better on the CORRECT alignment. If 'legacy' is lower,", rank)
    H.log("  stop and report — the library layout differs from what this script assumes.", rank)
    H.log("=" * 78, rank)
    return v


def main():
    cfg = H.load_cfg()
    world, rank, local, device = setup_dist()
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_rows = H.read_jsonl(cfg["train_jsonl"])
    val_rows = H.read_jsonl(cfg["val_jsonl"])
    spk = torch.load(os.path.join(out_dir, "speaker_embedding.pt"), map_location="cpu").float()

    t_load = time.time()
    core, hf_cfg = load_core(cfg, device)
    H.log(f"world={world} device={device} model loaded in {time.time()-t_load:.0f}s", rank)
    core.talker.config.use_cache = False
    core.talker.code_predictor.config.use_cache = False

    torch.manual_seed(cfg["seed"])  # identical LoRA init on every rank
    groups = H.lora_groups_from_cfg(cfg)
    counts = H.inject_lora(core, groups, cfg["lora_dropout"])
    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in core.parameters())
    H.log(f"LoRA layers {counts}; trainable {n_train/1e6:.2f}M / {n_all/1e6:.0f}M "
          f"({100*n_train/n_all:.2f}%) — base weights, embeddings, codec_head all frozen", rank)

    latest_path = os.path.join(out_dir, "adapter_latest.pt")
    best_path = os.path.join(out_dir, "adapter_best.pt")
    start_step, best_val = 0, float("inf")
    if cfg.get("resume") and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location="cpu")
        H.load_lora_state_dict(core, ck["lora"])
        start_step = int(ck.get("step", 0))
        best_val = float(ck.get("best_val", float("inf")))
        H.log(f"resumed adapter weights from {latest_path} at step {start_step} (optimizer state is reset)", rank)

    core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    core.train()

    wrapper = H.TrainWrapper(core, spk, cfg["sub_talker_loss_weight"]).to(device)
    if world > 1:
        frozen = [f"core.{n}" for n, p in core.named_parameters() if not p.requires_grad]
        DDP._set_params_and_buffers_to_ignore_for_model(wrapper, frozen)
        model = DDP(wrapper, device_ids=[local] if device.type == "cuda" else None,
                    broadcast_buffers=False, find_unused_parameters=False)
    else:
        model = wrapper

    collate = H.TTSCollator(hf_cfg)
    bs, accum = cfg["per_gpu_batch_size"], cfg["grad_accum"]
    tr_sampler = H.LengthGroupedBatchSampler([H.seq_len_of(r) for r in train_rows], bs, world, rank, cfg["seed"])
    va_sampler = H.LengthGroupedBatchSampler([H.seq_len_of(r) for r in val_rows], bs, world, rank, 0, shuffle=False)
    nw = cfg["num_workers"]
    tr_loader = DataLoader(H.HindiTTSDataset(train_rows), batch_sampler=tr_sampler, collate_fn=collate,
                           num_workers=nw, pin_memory=device.type == "cuda")
    va_loader = DataLoader(H.HindiTTSDataset(val_rows), batch_sampler=va_sampler, collate_fn=collate,
                           num_workers=nw, pin_memory=device.type == "cuda")

    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.99), eps=1e-8, weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, len(tr_sampler) // accum)
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup = max(10, int(total_steps * cfg["warmup_ratio"]))
    min_ratio = cfg["min_lr_ratio"]

    def lr_lambda(s):
        if s < warmup:
            return (s + 1) / warmup
        prog = min(1.0, (s - warmup) / max(1, total_steps - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    for _ in range(start_step):
        sched.step()
    H.log(f"train {len(train_rows)} utts | val {len(val_rows)} | per-GPU bs {bs} x accum {accum} x {world} GPU "
          f"= {bs*accum*world} utts/step | {steps_per_epoch} steps/epoch | {total_steps} total | warmup {warmup}", rank)

    if start_step == 0 and cfg["diagnostic_batches"] > 0:
        alignment_diagnostic(wrapper, va_loader, device, rank, cfg["diagnostic_batches"])
        base_val = evaluate(wrapper, va_loader, device)
        H.log(f"[val @ step 0] {json.dumps(base_val)}", rank)

    metrics_path = os.path.join(out_dir, "train_metrics.jsonl")
    mf = open(metrics_path, "a", encoding="utf-8") if rank == 0 else None

    def record(d):
        if mf:
            mf.write(json.dumps(d) + "\n")
            mf.flush()

    step = start_step
    start_epoch = start_step // steps_per_epoch
    skip_micro = (start_step % steps_per_epoch) * accum
    patience, stop, reason = 0, False, "completed all epochs"
    run = torch.zeros(5, device=device, dtype=torch.float64)
    t_last, tok_count = time.time(), 0
    for epoch in range(start_epoch, cfg["num_epochs"]):
        tr_sampler.set_epoch(epoch)
        for bi, batch in enumerate(tr_loader):
            if skip_micro > 0:
                skip_micro -= 1
                continue
            batch = H.move_batch(batch, device)
            sync_now = (bi + 1) % accum == 0
            ctx = model.no_sync() if (world > 1 and not sync_now) else contextlib.nullcontext()
            with ctx:
                loss, st = model(batch)
                (loss / accum).backward()
            run += torch.stack([st["main"].double(), st["sub"].double(), st["main_acc"].double(),
                                st["sub_acc"].double(), torch.ones((), device=device, dtype=torch.float64)])
            tok_count += int(batch["attn"].sum())
            if not sync_now:
                continue

            gn = torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            if not torch.isfinite(gn):
                H.log(f"non-finite grad norm at step {step}; skipping update", rank)
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step == 1 and device.type == "cuda":
                H.log(f"peak GPU memory after first step: {torch.cuda.max_memory_allocated(device)/2**30:.2f} GiB", rank)
            if step % cfg["log_every"] == 0:
                r = all_reduce(run.clone())
                k = r[4].clamp_min(1)
                dt = time.time() - t_last
                rec = {"step": step, "epoch": epoch, "lr": sched.get_last_lr()[0], "gn": float(gn),
                       "main": (r[0] / k).item(), "sub": (r[1] / k).item(),
                       "main_acc": (r[2] / k).item(), "sub_acc": (r[3] / k).item(),
                       "tok_per_s": tok_count * world / max(dt, 1e-6),
                       "min_left": H.seconds_left(cfg) / 60}
                H.log(f"ep {epoch} step {step}/{total_steps} | c0 {rec['main']:.3f} (acc {rec['main_acc']:.3f}) "
                      f"| sub {rec['sub']:.3f} (acc {rec['sub_acc']:.3f}) | lr {rec['lr']:.2e} | gn {rec['gn']:.2f} "
                      f"| {rec['tok_per_s']:.0f} tok/s | {rec['min_left']:.0f} min left", rank)
                record({"type": "train", **rec})
                run.zero_()
                t_last, tok_count = time.time(), 0

            if step % cfg["eval_every"] == 0:
                v = evaluate(wrapper, va_loader, device)
                improved = v["total"] < best_val - 1e-4
                if improved:
                    best_val, patience = v["total"], 0
                else:
                    patience += 1
                H.log(f"[val @ step {step}] total {v['total']:.4f} c0 {v['main']:.4f} (acc {v['main_acc']:.3f}) "
                      f"sub {v['sub']:.4f} | best {best_val:.4f} {'*NEW BEST*' if improved else f'(patience {patience})'}", rank)
                record({"type": "val", "step": step, **v})
                if rank == 0:
                    meta = {"step": step, "val": v, "best_val": best_val, "groups": groups}
                    save_adapter(core, latest_path, meta)
                    if improved:
                        save_adapter(core, best_path, meta)
                if patience >= cfg["early_stop_patience"]:
                    stop, reason = True, f"early stop (no val improvement for {patience} evals)"

            flag = torch.tensor([1.0 if H.seconds_left(cfg) <= 0 else 0.0], device=device)
            all_reduce(flag, op=dist.ReduceOp.MAX if dist.is_initialized() else None)
            if flag.item() > 0 and not stop:
                stop, reason = True, "time budget reached"
            if stop:
                break
        opt.zero_grad(set_to_none=True)   # drop partial accumulation at epoch end
        if stop:
            break

    v = evaluate(wrapper, va_loader, device)
    if v["total"] < best_val - 1e-4:
        best_val = v["total"]
        if rank == 0:
            save_adapter(core, best_path, {"step": step, "val": v, "best_val": best_val, "groups": groups})
    if rank == 0:
        save_adapter(core, latest_path, {"step": step, "val": v, "best_val": best_val, "groups": groups})
        if not os.path.exists(best_path):
            save_adapter(core, best_path, {"step": step, "val": v, "best_val": best_val, "groups": groups})
        with open(os.path.join(out_dir, "train_summary.json"), "w") as f:
            json.dump({"stopped": reason, "steps": step, "final_val": v, "best_val": best_val,
                       "lora_groups": groups, "trainable_params": n_train}, f, indent=2)
        record({"type": "final", "step": step, **v})
    with torch.no_grad():
        csum = torch.tensor([sum(float(p.double().sum()) for p in params)], device=device, dtype=torch.float64)
    mx, mn = all_reduce(csum.clone(), op=dist.ReduceOp.MAX if dist.is_initialized() else None), \
        -all_reduce(-csum.clone(), op=dist.ReduceOp.MAX if dist.is_initialized() else None)
    H.log(f"rank-sync check: LoRA checksum max-min = {(mx - mn).item():.3e} (must be ~0)", rank)
    H.log(f"training finished: {reason}; steps={step}; best val={best_val:.4f}", rank)
    if mf:
        mf.close()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
