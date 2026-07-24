"""Multi-seed comparison of shared vs no-shared MoE, with the *fair* causal test.

For each checkpoint:
  - soft (gate-weighted) NMI(Expert;Domain), mean over layers
  - conditional NMI I(E;D|Token) at a representative layer (token-identity control)
  - WHOLE-BRANCH ablation: ΔPPL from removing the entire routed branch vs the
    entire shared branch (like-for-like, unlike single-expert ablation)

Aggregates by condition (mean±std over seeds) and runs an exact permutation test
on soft NMI between the two conditions.

Usage:
    python -m analysis.seeds_experiment \
        --shared   A/step_3000,B/step_3000,C/step_3000 \
        --noshared D/step_3000,E/step_3000,F/step_3000 \
        --tokens-per-domain 80000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model, AnalysisTransformer
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.run_analysis import batched, domain_ppl
from analysis import metrics as M


@torch.inference_mode()
def capture_soft(model, eval_sets, device, bs, L):
    idx = [[] for _ in range(L)]; wt = [[] for _ in range(L)]; top1 = [[] for _ in range(L)]
    dom, tks = [], []
    for name in DOMAIN_NAMES:
        did = DOMAIN_NAMES.index(name)
        for ids, lab, pos in batched(eval_sets[name], bs):
            ids, pos = ids.to(device), pos.to(device)
            _, caps = model(ids, position_ids=pos, capture_routing=True)
            dom.append(np.full(ids.numel(), did, dtype=np.int64))
            tks.append(ids.reshape(-1).cpu().numpy())
            for li in range(L):
                ti = caps[li].topk_indices[0].numpy()      # [n,K]
                tw = caps[li].topk_weights[0].numpy()       # [n,K]
                idx[li].append(ti); wt[li].append(tw); top1[li].append(ti[:, 0])
    idx = [np.concatenate(x) for x in idx]
    wt = [np.concatenate(x) for x in wt]
    top1 = [np.concatenate(x) for x in top1]
    return idx, wt, top1, np.concatenate(dom), np.concatenate(tks)


def analyze(step_dir, eval_sets, device, bs):
    cfg = ModelConfig(**json.load(open(os.path.join(step_dir, "config.json"))))
    E, D, L = cfg.num_experts, len(DOMAIN_NAMES), cfg.num_layers
    mid = L // 2
    model = load_analysis_model(cfg, os.path.join(step_dir, "model.safetensors"), device)
    idx, wt, top1, dom, tks = capture_soft(model, eval_sets, device, bs, L)

    soft = [M.soft_nmi(idx[li], wt[li], dom, E, D) for li in range(L)]
    cond = M.conditional_nmi(top1[mid], dom, tks, E, D)

    base = np.mean([domain_ppl(model, eval_sets[n], device, bs) for n in DOMAIN_NAMES])
    d_routed = np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_routed=True)
                        for n in DOMAIN_NAMES]) - base
    d_shared = np.mean([domain_ppl(model, eval_sets[n], device, bs, ablate_shared=True)
                        for n in DOMAIN_NAMES]) - base
    return dict(soft_mean=float(np.mean(soft)), soft_by_layer=soft,
                cond=cond["conditional"], uncond=cond["unconditional"],
                base=float(base), d_routed=float(d_routed), d_shared=float(d_shared),
                has_shared=cfg.use_shared_expert)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared", required=True, help="comma list of step dirs")
    ap.add_argument("--noshared", required=True)
    ap.add_argument("--tokens-per-domain", type=int, default=80_000)
    ap.add_argument("--skip", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    any_cfg = ModelConfig(**json.load(open(os.path.join(args.shared.split(",")[0], "config.json"))))
    eval_sets = build_domain_eval(tok, any_cfg.max_seq_len, args.tokens_per_domain, args.skip)

    groups = {"shared": args.shared.split(","), "no-shared": args.noshared.split(",")}
    res = {k: [] for k in groups}
    print(f"\n{'condition':>10} {'seed_dir':>18} | {'softNMI':>8} {'condNMI':>8} {'uncNMI':>7} "
          f"| {'basePPL':>7} {'Δrm routed':>10} {'Δrm shared':>10}")
    print("-" * 92)
    for cond, dirs in groups.items():
        for d in dirs:
            r = analyze(d, eval_sets, device, args.batch)
            res[cond].append(r)
            tag = os.path.basename(os.path.dirname(d))
            print(f"{cond:>10} {tag:>18} | {r['soft_mean']:>8.4f} {r['cond']:>8.4f} {r['uncond']:>7.4f} "
                  f"| {r['base']:>7.2f} {r['d_routed']:>+10.2f} {r['d_shared']:>+10.2f}", flush=True)

    a = [r["soft_mean"] for r in res["shared"]]
    b = [r["soft_mean"] for r in res["no-shared"]]
    print("\n=== soft NMI(Expert;Domain), mean±std over seeds ===")
    print(f"  shared   : {np.mean(a):.4f} ± {np.std(a):.4f}   (n={len(a)})")
    print(f"  no-shared: {np.mean(b):.4f} ± {np.std(b):.4f}   (n={len(b)})")
    diff, p = M.permutation_test(b, a)   # no-shared minus shared
    print(f"  Δ(no-shared − shared) = {diff:+.4f}   exact permutation p = {p:.3f}")

    print("\n=== whole-branch ablation ΔPPL (mean±std over seeds) ===")
    for cond in groups:
        dr = [r["d_routed"] for r in res[cond]]; ds = [r["d_shared"] for r in res[cond]]
        print(f"  {cond:>10}: remove routed Δ={np.mean(dr):+.2f}±{np.std(dr):.2f} | "
              f"remove shared Δ={np.mean(ds):+.2f}±{np.std(ds):.2f}")


if __name__ == "__main__":
    main()
