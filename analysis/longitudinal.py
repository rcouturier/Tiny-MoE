"""Longitudinal emergence curve: replay NMI(Expert;Domain) + shared-expert
dominance across a directory of checkpoints (step_*). Builds the domain eval
ONCE and reuses it for every checkpoint.

Usage:
    python -m analysis.longitudinal --pilot ~/tiny_moe_study/pilot --tokens-per-domain 80000
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis import metrics as M
from analysis.run_analysis import batched, capture_routing, domain_ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--tokens-per-domain", type=int, default=80_000)
    ap.add_argument("--skip", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    steps = sorted(glob.glob(os.path.join(args.pilot, "step_*")),
                   key=lambda p: int(p.split("_")[-1]))
    cfg0 = ModelConfig(**json.load(open(os.path.join(steps[0], "config.json"))))
    E, D, L = cfg0.num_experts, len(DOMAIN_NAMES), cfg0.num_layers

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eval_sets = build_domain_eval(tok, cfg0.max_seq_len, args.tokens_per_domain, args.skip)

    print(f"\n{'step':>6} | {'mean NMI':>9} | {'perm ctl':>9} | {'shared ΔPPL':>11} | {'routed/shared':>13}")
    print("-" * 66)
    for sd in steps:
        step = int(sd.split("_")[-1])
        cfg = ModelConfig(**json.load(open(os.path.join(sd, "config.json"))))
        model = load_analysis_model(cfg, os.path.join(sd, "model.safetensors"), device)
        exp, dom, tks, snorm, rnorm = capture_routing(model, eval_sets, device, args.batch, L)
        nmis = [M.nmi(exp[li], dom, E, D) for li in range(L)]
        cm, _ = M.permutation_control(exp[L // 2], dom, E, D, n_perm=10)
        # shared-expert dominance (mean ΔPPL over domains)
        base = np.mean([domain_ppl(model, eval_sets[n], device, args.batch) for n in DOMAIN_NAMES])
        abl = np.mean([domain_ppl(model, eval_sets[n], device, args.batch, ablate_shared=True)
                       for n in DOMAIN_NAMES])
        ratio = np.mean([np.mean(np.concatenate(rnorm[n])) / max(1e-9, np.mean(np.concatenate(snorm[n])))
                         for n in DOMAIN_NAMES])
        print(f"{step:>6} | {np.mean(nmis):>9.4f} | {cm:>9.4f} | {abl-base:>+11.2f} | {ratio:>13.3f}",
              flush=True)


if __name__ == "__main__":
    main()
