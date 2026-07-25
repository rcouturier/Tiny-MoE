"""Unified per-group summary: soft NMI(Expert;Domain), load imbalance, base PPL,
and whole-branch ablation (remove routed / remove shared), with pairwise
permutation tests on soft NMI. Used for the D1xD2 interaction, the scale sweep,
and shared-dominance-vs-aux.

Usage:
    python -m analysis.summary --group name=d1,d2,... [--group ...] --tokens-per-domain 80000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.run_analysis import domain_ppl
from analysis.seeds_experiment import capture_soft
from analysis.d2_experiment import load_imbalance
from analysis import metrics as M


def analyze(step_dir, eval_sets, device, bs):
    cfg = ModelConfig(**json.load(open(os.path.join(step_dir, "config.json"))))
    E, D, L = cfg.num_experts, len(DOMAIN_NAMES), cfg.num_layers
    model = load_analysis_model(cfg, os.path.join(step_dir, "model.safetensors"), device)
    idx, wt, top1, dom, tks = capture_soft(model, eval_sets, device, bs, L)
    soft = float(np.mean([M.soft_nmi(idx[li], wt[li], dom, E, D) for li in range(L)]))
    base = float(np.mean([domain_ppl(model, eval_sets[n], device, bs) for n in DOMAIN_NAMES]))
    dr = float(np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_routed=True)
                        for n in DOMAIN_NAMES]) - base)
    ds = float(np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_shared=True)
                        for n in DOMAIN_NAMES]) - base)
    return dict(soft=soft, imb=load_imbalance(top1, E, L), base=base,
                d_routed=dr, d_shared=ds, has_shared=cfg.use_shared_expert)


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
    print(f"\n{'group':>12} {'seed_dir':>16} | {'softNMI':>8} {'loadImb':>8} {'basePPL':>7} "
          f"{'Δrmrouted':>10} {'Δrmshared':>10}")
    print("-" * 82)
    for name, dirs in groups.items():
        for d in dirs:
            r = analyze(d, eval_sets, device, args.batch)
            res[name].append(r)
            print(f"{name:>12} {os.path.basename(os.path.dirname(d)):>16} | {r['soft']:>8.4f} "
                  f"{r['imb']:>8.4f} {r['base']:>7.2f} {r['d_routed']:>+10.2f} {r['d_shared']:>+10.2f}",
                  flush=True)

    print("\n=== per group (mean±std over seeds) ===")
    for name in groups:
        g = res[name]
        f = lambda k: (np.mean([r[k] for r in g]), np.std([r[k] for r in g]))
        s, im, b, dr, ds = f("soft"), f("imb"), f("base"), f("d_routed"), f("d_shared")
        print(f"  {name:>12}: softNMI={s[0]:.4f}±{s[1]:.4f}  loadImb={im[0]:.4f}±{im[1]:.4f}  "
              f"PPL={b[0]:.2f}  Δrmrouted={dr[0]:+.2f}±{dr[1]:.2f}  Δrmshared={ds[0]:+.2f}±{ds[1]:.2f}")

    names = list(groups)
    if len(names) > 1:
        print("\n=== pairwise permutation tests on soft NMI ===")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = [r["soft"] for r in res[names[i]]]; b = [r["soft"] for r in res[names[j]]]
                diff, p = M.permutation_test(a, b)
                print(f"  {names[i]} vs {names[j]}: Δ={diff:+.4f}  p={p:.3f}")


if __name__ == "__main__":
    main()
