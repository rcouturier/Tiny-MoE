"""D2: load-balancing sweep. Does routed-expert domain specialization rise as the
aux-loss coefficient -> 0 (less balancing pressure)? Reports soft NMI AND expert
load imbalance per group, with pairwise permutation tests, to expose the
balance <-> specialization tradeoff.

Usage:
    python -m analysis.d2_experiment \
        --group aux0=A/step_3000,B/step_3000,... \
        --group aux1e-2=... --group aux1e-1=... --tokens-per-domain 80000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.run_analysis import batched, domain_ppl
from analysis.seeds_experiment import capture_soft
from analysis import metrics as M


def load_imbalance(top1, E, L):
    """Mean over layers of the std of top-1 expert load P(e). 0 = perfectly
    balanced; higher = more imbalanced (a prerequisite for specialization)."""
    vals = []
    for li in range(L):
        c = np.bincount(np.asarray(top1[li]).astype(np.int64), minlength=E).astype(float)
        vals.append((c / c.sum()).std())
    return float(np.mean(vals))


def analyze(step_dir, eval_sets, device, bs):
    cfg = ModelConfig(**json.load(open(os.path.join(step_dir, "config.json"))))
    E, D, L = cfg.num_experts, len(DOMAIN_NAMES), cfg.num_layers
    model = load_analysis_model(cfg, os.path.join(step_dir, "model.safetensors"), device)
    idx, wt, top1, dom, tks = capture_soft(model, eval_sets, device, bs, L)
    soft = np.mean([M.soft_nmi(idx[li], wt[li], dom, E, D) for li in range(L)])
    base = np.mean([domain_ppl(model, eval_sets[n], device, bs) for n in DOMAIN_NAMES])
    return dict(soft=float(soft), imb=load_imbalance(top1, E, L), base=float(base))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True, help="name=dir1,dir2,...")
    ap.add_argument("--tokens-per-domain", type=int, default=80_000)
    ap.add_argument("--skip", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    groups = {}
    for g in args.group:
        name, dirs = g.split("=", 1)
        groups[name] = dirs.split(",")

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    first = next(iter(groups.values()))[0]
    any_cfg = ModelConfig(**json.load(open(os.path.join(first, "config.json"))))
    eval_sets = build_domain_eval(tok, any_cfg.max_seq_len, args.tokens_per_domain, args.skip)

    res = {name: [] for name in groups}
    print(f"\n{'group':>10} {'seed_dir':>18} | {'softNMI':>8} {'loadImb':>8} {'basePPL':>7}")
    print("-" * 60)
    for name, dirs in groups.items():
        for d in dirs:
            r = analyze(d, eval_sets, device, args.batch)
            res[name].append(r)
            print(f"{name:>10} {os.path.basename(os.path.dirname(d)):>18} | "
                  f"{r['soft']:>8.4f} {r['imb']:>8.4f} {r['base']:>7.2f}", flush=True)

    print("\n=== per group: soft NMI and load imbalance (mean±std over seeds) ===")
    for name in groups:
        s = [r["soft"] for r in res[name]]; im = [r["imb"] for r in res[name]]
        b = [r["base"] for r in res[name]]
        print(f"  {name:>8}: softNMI={np.mean(s):.4f}±{np.std(s):.4f}  "
              f"loadImb={np.mean(im):.4f}±{np.std(im):.4f}  basePPL={np.mean(b):.2f}±{np.std(b):.2f}")

    names = list(groups)
    print("\n=== pairwise permutation tests on soft NMI ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = [r["soft"] for r in res[names[i]]]; b = [r["soft"] for r in res[names[j]]]
            diff, p = M.permutation_test(a, b)
            print(f"  {names[i]} vs {names[j]}: Δ={diff:+.4f}  p={p:.3f}")


if __name__ == "__main__":
    main()
