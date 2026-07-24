"""Standalone single-GPU pilot trainer (plain PyTorch, no DeepSpeed/accelerate/
bitsandbytes/wandb). Reuses the repo model and streaming data. Saves safetensors
checkpoints + a config.json at intervals for longitudinal analysis.

Usage:
    python -m analysis.pilot_train --steps 1500 --num-layers 8 --batch 24 \
        --save-every 500 --out /path/to/pilot
"""
from __future__ import annotations
import argparse, json, math, os, time, dataclasses

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from safetensors.torch import save_file

from training.training_configs import ModelConfig
from training.model_train import Transformer
from training.data import Training_Streaming_Dataset


def cosine_lr(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


def save_ckpt(model, config, out_dir, step):
    d = os.path.join(out_dir, f"step_{step}")
    os.makedirs(d, exist_ok=True)
    sd = model.state_dict()
    clean, seen = {}, {}
    for k, t in sd.items():
        k = k.replace("_orig_mod.", "").replace("module.", "")
        t = t.contiguous()
        p = t.data_ptr()
        clean[k] = t.clone() if p in seen else t
        seen[p] = k
    save_file(clean, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2)
    print(f"[ckpt] saved {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--aux-coef", type=float, default=1e-2)
    ap.add_argument("--z-coef", type=float, default=1e-3)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    config = ModelConfig(num_layers=args.num_layers, num_experts=args.num_experts,
                         max_seq_len=args.seq_len)
    model = Transformer(config).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.1f}M params | {args.num_layers} layers | {args.num_experts} experts")

    ds = Training_Streaming_Dataset(config, tok, seed=args.seed, shuffle_buffer_size=20_000)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=2, prefetch_factor=2,
                    persistent_workers=True, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    it = iter(dl)
    t0 = time.perf_counter()
    step = 0
    save_ckpt(model, config, args.out, 0)  # untrained floor for the longitudinal curve
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        agg = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            ids = batch["input_ids"].to(device, non_blocking=True)
            pos = batch["position_ids"].to(device, non_blocking=True)
            lab = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, position_ids=pos)
                ce = F.cross_entropy(out["logits"].reshape(-1, out["logits"].size(-1)),
                                     lab.reshape(-1), ignore_index=-100)
                loss = ce + args.aux_coef * out["aux_loss"] + args.z_coef * out["z_loss"]
            (loss / args.grad_accum).backward()
            agg += loss.item() / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.lr, args.warmup, args.steps)
        opt.step()
        step += 1
        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            tps = step * args.batch * args.grad_accum * args.seq_len / dt
            print(f"step {step:5d} | loss {agg:.4f} | ppl {math.exp(min(agg,20)):.2f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} | {tps/1e3:.1f}k tok/s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(model, config, args.out, step)
    print(f"[done] {step} steps in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
