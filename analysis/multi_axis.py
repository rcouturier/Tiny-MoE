"""Multi-axis specialization: along WHICH axis do routed experts specialize?
Compares soft NMI(Expert; axis) for axis in {domain, surface-class, token-
frequency bucket, sequence-position bucket}, vs a permutation control. Run on the
aux=0 models where specialization is strongest.

Usage:
    python -m analysis.multi_axis --ckpts A/step_3000,B/step_3000 --tokens-per-domain 80000
"""
from __future__ import annotations
import argparse, json, os, string
import numpy as np
import torch
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis.run_analysis import batched
from analysis import metrics as M

N_FREQ_BUCKETS = 5
N_POS_BUCKETS = 4
SURFACE_CLASSES = ["word", "number", "punct", "space", "mixed"]


def build_surface_map(tokenizer, vocab_size):
    """Map each vocab id -> surface class index (lexical proxy for syntax)."""
    punct = set(string.punctuation)
    cls = np.zeros(vocab_size, dtype=np.int64)
    for i in range(vocab_size):
        tok = tokenizer.convert_ids_to_tokens(i)
        if tok is None:
            cls[i] = 4; continue
        core = tok.replace("▁", "").strip()  # strip Mistral's '▁' space marker
        if core == "":
            cls[i] = 3
        elif core.isdigit():
            cls[i] = 1
        elif all(c in punct for c in core):
            cls[i] = 2
        elif core.isalpha():
            cls[i] = 0
        else:
            cls[i] = 4
    return cls


@torch.inference_mode()
def capture(model, eval_sets, device, bs, L, seq_len):
    idx = [[] for _ in range(L)]; wt = [[] for _ in range(L)]
    dom, tks, pos = [], [], []
    for name in DOMAIN_NAMES:
        did = DOMAIN_NAMES.index(name)
        for ids, lab, p in batched(eval_sets[name], bs):
            B = ids.shape[0]
            ids_d, p_d = ids.to(device), p.to(device)
            _, caps = model(ids_d, position_ids=p_d, capture_routing=True)
            dom.append(np.full(ids.numel(), did, dtype=np.int64))
            tks.append(ids.reshape(-1).numpy())
            pos.append(np.tile(np.arange(seq_len), B))
            for li in range(L):
                idx[li].append(caps[li].topk_indices[0].numpy())
                wt[li].append(caps[li].topk_weights[0].numpy())
    idx = [np.concatenate(x) for x in idx]; wt = [np.concatenate(x) for x in wt]
    return idx, wt, np.concatenate(dom), np.concatenate(tks), np.concatenate(pos)


def analyze(step_dir, eval_sets, surface_map, device, bs):
    cfg = ModelConfig(**json.load(open(os.path.join(step_dir, "config.json"))))
    E, L = cfg.num_experts, cfg.num_layers
    model = load_analysis_model(cfg, os.path.join(step_dir, "model.safetensors"), device)
    idx, wt, dom, tks, pos = capture(model, eval_sets, device, bs, L, cfg.max_seq_len)

    # axis label vectors
    surface = surface_map[tks]
    # frequency buckets by token rank
    uniq, inv, counts = np.unique(tks, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    tok_rank = rank[inv]
    freq_bucket = np.minimum((tok_rank * N_FREQ_BUCKETS) // len(uniq), N_FREQ_BUCKETS - 1)
    pos_bucket = np.minimum((pos * N_POS_BUCKETS) // cfg.max_seq_len, N_POS_BUCKETS - 1)

    axes = {"domain": (dom, len(DOMAIN_NAMES)), "surface": (surface, len(SURFACE_CLASSES)),
            "frequency": (freq_bucket, N_FREQ_BUCKETS), "position": (pos_bucket, N_POS_BUCKETS)}
    out = {}
    for aname, (labels, ncls) in axes.items():
        nmi = float(np.mean([M.soft_nmi(idx[li], wt[li], labels, E, ncls) for li in range(L)]))
        cm, _ = M.permutation_control(idx[L // 2][:, 0], labels, E, ncls, n_perm=10)
        out[aname] = (nmi, cm)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma list of step dirs")
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
    surface_map = build_surface_map(tok, cfg0.vocab_size)
    eval_sets = build_domain_eval(tok, cfg0.max_seq_len, args.tokens_per_domain, args.skip)

    agg = {a: [] for a in ["domain", "surface", "frequency", "position"]}
    ctl = {a: [] for a in agg}
    for d in dirs:
        r = analyze(d, eval_sets, surface_map, device, args.batch)
        tag = os.path.basename(os.path.dirname(d))
        print(f"  {tag:>10}: " + "  ".join(f"{a}={r[a][0]:.4f}" for a in agg), flush=True)
        for a in agg:
            agg[a].append(r[a][0]); ctl[a].append(r[a][1])

    print("\n=== soft NMI(Expert; axis), mean±std over checkpoints (control in parens) ===")
    for a in sorted(agg, key=lambda k: -np.mean(agg[k])):
        print(f"  {a:>10}: {np.mean(agg[a]):.4f} ± {np.std(agg[a]):.4f}   (control {np.mean(ctl[a]):.4f})")


if __name__ == "__main__":
    main()
