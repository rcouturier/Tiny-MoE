"""Axis disentanglement: are domain/surface/frequency specialization independent,
or does one explain the others? Computes NMI(Expert; X) and NMI(Expert; X | Y)
for X,Y in {domain, surface, frequency}. If NMI(E;domain|surface) collapses while
NMI(E;surface|domain) stays high, surface form is the primary organizing axis and
"domain specialization" is largely a correlate of it.

Averages over the given checkpoints (use the aux=0 seeds where signal is strong).

Usage:
    python -m analysis.disentangle --ckpts A/step_3000,B/step_3000,... --tokens-per-domain 80000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.multi_axis import capture, build_surface_map, N_FREQ_BUCKETS, SURFACE_CLASSES
from analysis import metrics as M


def make_labels(dom, tks, cfg):
    surface = SURFACE_MAP[tks]
    uniq, inv, counts = np.unique(tks, return_inverse=True, return_counts=True)
    order = np.argsort(-counts); rank = np.empty_like(order); rank[order] = np.arange(len(order))
    freq = np.minimum((rank[inv] * N_FREQ_BUCKETS) // len(uniq), N_FREQ_BUCKETS - 1)
    return {"domain": (dom, len(DOMAIN_NAMES)), "surface": (surface, len(SURFACE_CLASSES)),
            "frequency": (freq, N_FREQ_BUCKETS)}


def analyze(step_dir, eval_sets, device, bs):
    cfg = ModelConfig(**json.load(open(os.path.join(step_dir, "config.json"))))
    E, L = cfg.num_experts, cfg.num_layers
    model = load_analysis_model(cfg, os.path.join(step_dir, "model.safetensors"), device)
    idx, wt, dom, tks, pos = capture(model, eval_sets, device, bs, L, cfg.max_seq_len)
    labels = make_labels(dom, tks, cfg)

    pairs = [("domain", "surface"), ("domain", "frequency"),
             ("surface", "domain"), ("surface", "frequency"),
             ("frequency", "domain"), ("frequency", "surface")]
    # average over layers (top-1 expert)
    res = {}
    for tgt, cond in pairs:
        tv, nt = labels[tgt]; cv, nc = labels[cond]
        vals = [M.conditional_nmi_general(idx[li][:, 0], tv, cv, E, nt, nc) for li in range(L)]
        res[(tgt, cond)] = (np.mean([v["uncond"] for v in vals]),
                            np.mean([v["cond"] for v in vals]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--tokens-per-domain", type=int, default=80_000)
    ap.add_argument("--skip", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    dirs = args.ckpts.split(",")
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg0 = ModelConfig(**json.load(open(os.path.join(dirs[0], "config.json"))))
    global SURFACE_MAP
    SURFACE_MAP = build_surface_map(tok, cfg0.vocab_size)
    eval_sets = build_domain_eval(tok, cfg0.max_seq_len, args.tokens_per_domain, args.skip)

    acc = {}
    for d in dirs:
        r = analyze(d, eval_sets, device, args.batch)
        for k, v in r.items():
            acc.setdefault(k, []).append(v)

    print("\n=== NMI(Expert; target) unconditional  vs  conditional on another axis ===")
    print(f"{'target':>10} | {'conditioned on':>14} | {'uncond':>7} | {'cond':>7} | {'retained %':>10}")
    print("-" * 62)
    for (tgt, cond), vals in acc.items():
        u = np.mean([v[0] for v in vals]); c = np.mean([v[1] for v in vals])
        print(f"{tgt:>10} | {cond:>14} | {u:>7.4f} | {c:>7.4f} | {100*c/max(u,1e-9):>9.0f}%")


if __name__ == "__main__":
    main()
