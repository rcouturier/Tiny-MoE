"""On-the-fly knowledge distillation from a Mistral-7B teacher into the Tiny-MoE
student (same tokenizer/vocab). Combines KD (KL to teacher soft targets) with the
usual next-token CE and the MoE stability losses. Saves checkpoints in the same
format as pilot_train so the analysis toolkit reads them unchanged.

Usage:
    python -m analysis.distill_train --steps 12000 --batch 24 --kd-alpha 0.9 \
        --kd-temp 2.0 --out /path/to/distill
"""
from __future__ import annotations
import argparse, math, os, time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from training.training_configs import ModelConfig
from training.model_train import Transformer
from training.data import Training_Streaming_Dataset
from analysis.pilot_train import cosine_lr, save_ckpt

TEACHER = "mistralai/Mistral-7B-v0.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=800)
    ap.add_argument("--aux-coef", type=float, default=1e-2)
    ap.add_argument("--z-coef", type=float, default=1e-3)
    ap.add_argument("--kd-alpha", type=float, default=0.9, help="weight on KD vs CE")
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument("--shared-dropout", type=float, default=0.0)
    ap.add_argument("--save-every", type=int, default=3000)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(TEACHER)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[teacher] loading Mistral-7B ...", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.bfloat16).to(device).eval()
    teacher.requires_grad_(False)

    config = ModelConfig(num_layers=args.num_layers, num_experts=args.num_experts,
                         num_experts_per_token=args.top_k, max_seq_len=args.seq_len,
                         use_shared_expert=not args.no_shared, shared_dropout=args.shared_dropout)
    student = Transformer(config).to(device).train()
    print(f"[student] {sum(p.numel() for p in student.parameters())/1e6:.1f}M params | "
          f"KD α={args.kd_alpha} τ={args.kd_temp}", flush=True)

    ds = Training_Streaming_Dataset(config, tok, seed=args.seed, shuffle_buffer_size=20_000)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=2, prefetch_factor=2,
                    persistent_workers=True, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    T = args.kd_temp
    it = iter(dl)
    t0 = time.perf_counter()
    step = 0
    save_ckpt(student, config, args.out, 0)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        ids = batch["input_ids"].to(device, non_blocking=True)
        pos = batch["position_ids"].to(device, non_blocking=True)
        lab = batch["labels"].to(device, non_blocking=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            t_logits = teacher(ids).logits  # [B,T,V]

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student(ids, position_ids=pos)
            s_logits = out["logits"]
            ce = F.cross_entropy(s_logits.reshape(-1, s_logits.size(-1)), lab.reshape(-1),
                                 ignore_index=-100)
        # KD in fp32 for stability
        s_logp = F.log_softmax(s_logits.float() / T, dim=-1)
        t_prob = F.softmax(t_logits.float() / T, dim=-1)
        kd = F.kl_div(s_logp.reshape(-1, s_logp.size(-1)), t_prob.reshape(-1, t_prob.size(-1)),
                      reduction="batchmean") * (T * T)
        loss = (args.kd_alpha * kd + (1 - args.kd_alpha) * ce
                + args.aux_coef * out["aux_loss"] + args.z_coef * out["z_loss"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.lr, args.warmup, args.steps)
        opt.step()
        step += 1
        if step % args.log_every == 0:
            dt = time.perf_counter() - t0
            tps = step * args.batch * args.seq_len / dt
            print(f"step {step:5d} | loss {loss.item():.4f} | ce {ce.item():.4f} | kd {kd.item():.4f} "
                  f"| ppl {math.exp(min(ce.item(),20)):.2f} | {tps/1e3:.1f}k tok/s", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            save_ckpt(student, config, args.out, step)
    print(f"[done] {step} steps in {(time.perf_counter()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
