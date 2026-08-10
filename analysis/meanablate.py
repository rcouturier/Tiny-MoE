"""Zero- vs mean-ablation, side by side, plus the branch norm share.

Answers the measurement-validity objection to the study's headline statistic:
zero-ablating a branch in a residual stream perturbs the network off-distribution
by an amount that scales with the removed branch's output norm, and both headline
levers (branch width, shared LR) modulate that norm directly. Here each ablated
branch is instead replaced by its per-layer MEAN output over the eval stream
(computed in a capture pass), which removes the token-dependent computation while
approximately preserving the branch's average contribution to the residual stream.
If the shared share `s` ranks configurations the same way under both definitions,
the ordering is not a norm artifact.

Also reports the norm share ||f_sh|| / (||f_sh|| + ||routed||), token-averaged then
layer-averaged: the quantity the objection says `s` secretly measures.

Usage:
    python -m analysis.meanablate --runs A,B --steps last
    python -m analysis.meanablate --runs A,B --steps all      # every common checkpoint
    python -m analysis.meanablate --runs A --steps 12000,30000
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.run_analysis import domain_ppl, batched


@torch.inference_mode()
def branch_means_and_norms(model, eval_sets, device, bs, num_layers):
    """One capture pass over all domains: per-layer mean branch outputs + norm share."""
    L = num_layers
    ssum = [None] * L
    rsum = [None] * L
    ntok = 0
    snorm_tot, rnorm_tot = 0.0, 0.0
    for name in DOMAIN_NAMES:
        for ids, lab, pos in batched(eval_sets[name], bs):
            ids, pos = ids.to(device), pos.to(device)
            _, caps = model(ids, position_ids=pos, capture_routing=True)
            for li in range(L):
                sh = caps[li].shared_sum[0]
                rt = caps[li].routed_sum[0]
                ssum[li] = sh if ssum[li] is None else ssum[li] + sh
                rsum[li] = rt if rsum[li] is None else rsum[li] + rt
                snorm_tot += float(caps[li].shared_norm[0].sum())
                rnorm_tot += float(caps[li].routed_norm[0].sum())
            ntok += ids.numel()
    shared_subs = [(v / ntok).to(device) for v in ssum]
    routed_subs = [(v / ntok).to(device) for v in rsum]
    norm_share = snorm_tot / max(1e-9, snorm_tot + rnorm_tot)
    return shared_subs, routed_subs, norm_share


def analyze_step(step_dir, eval_sets, device, bs):
    cfg = ModelConfig(**json.load(open(os.path.join(step_dir, "config.json"))))
    model = load_analysis_model(cfg, os.path.join(step_dir, "model.safetensors"), device)
    L = cfg.num_layers
    shared_subs, routed_subs, norm_share = branch_means_and_norms(model, eval_sets, device, bs, L)
    base = float(np.mean([domain_ppl(model, eval_sets[n], device, bs) for n in DOMAIN_NAMES]))
    dz_sh = float(np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_shared=True)
                           for n in DOMAIN_NAMES]) - base)
    dz_rt = float(np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_routed=True)
                           for n in DOMAIN_NAMES]) - base)
    dm_sh = float(np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_shared=True,
                                      shared_subs=shared_subs) for n in DOMAIN_NAMES]) - base)
    dm_rt = float(np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_routed=True,
                                      routed_subs=routed_subs) for n in DOMAIN_NAMES]) - base)
    return dict(base=base, dz_sh=dz_sh, dz_rt=dz_rt, dm_sh=dm_sh, dm_rt=dm_rt,
                norm_share=norm_share)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of run dirs (seeds)")
    ap.add_argument("--steps", default="last", help="'last', 'all', or comma list of step values")
    ap.add_argument("--tokens-per-domain", type=int, default=60_000)
    ap.add_argument("--skip", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    runs = args.runs.split(",")
    stepsets = [{int(os.path.basename(d).split("_")[1]) for d in glob.glob(os.path.join(r, "step_*"))}
                for r in runs]
    common = sorted(set.intersection(*stepsets))
    if args.steps == "last":
        steps = [common[-1]]
    elif args.steps == "all":
        steps = common
    else:
        steps = [int(x) for x in args.steps.split(",")]

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    probe = os.path.join(runs[0], f"step_{steps[-1]}")
    cfg = ModelConfig(**json.load(open(os.path.join(probe, "config.json"))))
    eval_sets = build_domain_eval(tok, cfg.max_seq_len, args.tokens_per_domain, args.skip)

    print(f"\n{'step':>6} | {'base':>5} | {'Δsh_zero':>8} {'Δrt_zero':>8} {'s_zero':>6} "
          f"| {'Δsh_mean':>8} {'Δrt_mean':>8} {'s_mean':>6} | {'normshare':>9}")
    print("-" * 96)
    for st in steps:
        rows = [analyze_step(os.path.join(r, f"step_{st}"), eval_sets, device, args.batch)
                for r in runs]
        f = lambda k: float(np.mean([x[k] for x in rows]))
        sz = f("dz_sh") / max(1e-9, f("dz_sh") + f("dz_rt"))
        sm = f("dm_sh") / max(1e-9, f("dm_sh") + f("dm_rt"))
        print(f"{st:>6} | {f('base'):>5.2f} | {f('dz_sh'):>+8.2f} {f('dz_rt'):>+8.2f} {sz:>6.2f} "
              f"| {f('dm_sh'):>+8.2f} {f('dm_rt'):>+8.2f} {sm:>6.2f} | {f('norm_share'):>9.3f}",
              flush=True)


if __name__ == "__main__":
    main()
