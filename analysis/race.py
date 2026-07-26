"""The shared-vs-routed 'race': trajectory of branch importance over training.

For each checkpoint step (averaged over the given run dirs / seeds), reports base
PPL, ΔPPL from removing the shared branch, ΔPPL from removing the whole routed
branch, and routed soft NMI. If remove-shared grows while remove-routed stays flat,
the shared expert is progressively monopolizing the computation.

Usage:
    python -m analysis.race --runs A,B,C --tokens-per-domain 60000
    # A,B,C are run dirs each containing step_* subdirs at matching step values
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.summary import analyze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run dirs (each has step_* subdirs)")
    ap.add_argument("--tokens-per-domain", type=int, default=60_000)
    ap.add_argument("--skip", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    runs = args.runs.split(",")
    # steps present in ALL runs
    stepsets = []
    for r in runs:
        s = {int(os.path.basename(d).split("_")[1])
             for d in glob.glob(os.path.join(r, "step_*"))}
        stepsets.append(s)
    steps = sorted(set.intersection(*stepsets))

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probe = os.path.join(runs[0], f"step_{steps[-1]}")
    cfg = ModelConfig(**json.load(open(os.path.join(probe, "config.json"))))
    eval_sets = build_domain_eval(tok, cfg.max_seq_len, args.tokens_per_domain, args.skip)

    print(f"\n{'step':>6} | {'basePPL':>7} | {'Δrm shared':>10} | {'Δrm routed':>10} "
          f"| {'routed NMI':>10} | {'shared/(sh+rt)':>13}")
    print("-" * 74)
    for st in steps:
        rows = [analyze(os.path.join(r, f"step_{st}"), eval_sets, device, args.batch) for r in runs]
        f = lambda k: (np.mean([x[k] for x in rows]), np.std([x[k] for x in rows]))
        base, ds, dr, nmi = f("base"), f("d_shared"), f("d_routed"), f("soft")
        share = ds[0] / max(1e-9, ds[0] + dr[0])   # fraction of ablation-impact from shared
        print(f"{st:>6} | {base[0]:>7.2f} | {ds[0]:>+7.2f}±{ds[1]:.2f} | {dr[0]:>+7.2f}±{dr[1]:.2f} "
              f"| {nmi[0]:>10.4f} | {share:>13.2f}", flush=True)


if __name__ == "__main__":
    main()
