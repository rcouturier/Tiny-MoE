"""Run the full specialization analysis on a checkpoint:
observational (NMI/purity/token-context) + causal (expert & shared ablation ΔPPL).

Usage:
    python -m analysis.run_analysis --ckpt PILOT/step_1500/model.safetensors \
        --config PILOT/step_1500/config.json --tokens-per-domain 150000 --ablate-layer 4
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from training.training_configs import ModelConfig
from analysis.model_analysis import load_analysis_model
from analysis.domain_eval import build_domain_eval, DOMAIN_NAMES
from analysis import metrics as M


def batched(seqs, bs):
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        yield (torch.stack([s["input_ids"] for s in chunk]),
               torch.stack([s["labels"] for s in chunk]),
               torch.stack([s["position_ids"] for s in chunk]))


@torch.inference_mode()
def domain_ppl(model, seqs, device, bs, ablate_layer_experts=None, ablate_shared=False,
               ablate_routed=False):
    tot_loss, tot_tok = 0.0, 0
    for ids, lab, pos in batched(seqs, bs):
        ids, lab, pos = ids.to(device), lab.to(device), pos.to(device)
        logits, _ = model(ids, position_ids=pos, ablate_layer_experts=ablate_layer_experts,
                          ablate_shared=ablate_shared, ablate_routed=ablate_routed)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1),
                               reduction="sum")
        tot_loss += loss.item(); tot_tok += lab.numel()
    return tot_loss / tot_tok


@torch.inference_mode()
def capture_routing(model, eval_sets, device, bs, num_layers):
    """Returns per-layer arrays: expert_top1[L][Ntok], and shared domain/token arrays."""
    L = num_layers
    exp = [[] for _ in range(L)]
    dom, tks = [], []
    snorm = {n: [] for n in DOMAIN_NAMES}; rnorm = {n: [] for n in DOMAIN_NAMES}
    for name in DOMAIN_NAMES:
        did = DOMAIN_NAMES.index(name)
        for ids, lab, pos in batched(eval_sets[name], bs):
            ids, pos = ids.to(device), pos.to(device)
            _, caps = model(ids, position_ids=pos, capture_routing=True)
            n = ids.numel()
            dom.append(np.full(n, did, dtype=np.int64))
            tks.append(ids.reshape(-1).cpu().numpy())
            for li in range(L):
                exp[li].append(caps[li].topk_indices[0][:, 0].numpy())  # top-1
                snorm[name].append(caps[li].shared_norm[0].numpy())
                rnorm[name].append(caps[li].routed_norm[0].numpy())
    exp = [np.concatenate(e) for e in exp]
    dom = np.concatenate(dom); tks = np.concatenate(tks)
    return exp, dom, tks, snorm, rnorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens-per-domain", type=int, default=150_000)
    ap.add_argument("--skip", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ablate-layer", type=int, default=None,
                    help="layer index for the causal expert-ablation matrix (default: middle)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    cfg = ModelConfig(**json.load(open(args.config)))
    E, D, L = cfg.num_experts, len(DOMAIN_NAMES), cfg.num_layers
    ablate_layer = args.ablate_layer if args.ablate_layer is not None else L // 2

    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_analysis_model(cfg, args.ckpt, device)
    eval_sets = build_domain_eval(tok, cfg.max_seq_len, args.tokens_per_domain, args.skip)

    # ---------- Observational ----------
    exp, dom, tks, snorm, rnorm = capture_routing(model, eval_sets, device, args.batch, L)

    print("\n=== NMI(Expert; Domain) per layer  vs  permutation control ===")
    nmi_layer = []
    for li in range(L):
        nm = M.nmi(exp[li], dom, E, D)
        cm, cs = M.permutation_control(exp[li], dom, E, D, n_perm=20)
        nmi_layer.append(nm)
        flag = "  <-- signal" if nm > cm + 3 * cs else ""
        print(f"  layer {li:2d}: NMI={nm:.4f}   control={cm:.4f}±{cs:.4f}{flag}")
    print(f"  mean NMI over layers: {np.mean(nmi_layer):.4f}")

    print(f"\n=== Expert purity @ layer {ablate_layer} (dominant domain, share, entropy) ===")
    for p in M.purity(exp[ablate_layer], dom, E, D, DOMAIN_NAMES):
        if p["n"]:
            print(f"  expert {p['expert']}: {p['dominant']:4s}  share={p['share']:.2f}  "
                  f"H={p['entropy']:.2f}  (n={p['n']})")

    print(f"\n=== Q2 token-identity control @ layer {ablate_layer} ===")
    tc = M.token_context_divergence(exp[ablate_layer], dom, tks, E, D)
    print(f"  tokens used={tc['n_tokens']}  observed JSD={tc['observed_jsd']:.4f}  "
          f"control JSD={tc['control_jsd']:.4f}  "
          f"({'context-driven' if tc['observed_jsd'] > 2*tc['control_jsd'] else 'token-driven'})")

    print("\n=== Shared vs routed magnitude (mean ||routed|| / ||shared||) per domain ===")
    for name in DOMAIN_NAMES:
        s = np.concatenate(snorm[name]); r = np.concatenate(rnorm[name])
        print(f"  {name:4s}: {float(np.mean(r)/max(1e-9,np.mean(s))):.3f}")

    # ---------- Causal ----------
    print(f"\n=== Causal expert ablation @ layer {ablate_layer}: ΔPPL per domain ===")
    base = {n: domain_ppl(model, eval_sets[n], device, args.batch) for n in DOMAIN_NAMES}
    print("  baseline PPL: " + "  ".join(f"{n}={base[n]:.2f}" for n in DOMAIN_NAMES))
    R = M.routing_matrix(exp[ablate_layer], dom, E, D)     # [E, D] P(e|d)
    C = np.zeros((E, D))
    for e in range(E):
        for di, n in enumerate(DOMAIN_NAMES):
            ppl = domain_ppl(model, eval_sets[n], device, args.batch,
                             ablate_layer_experts={ablate_layer: {e}})
            C[e, di] = ppl - base[n]
        print(f"  ablate expert {e}: " +
              "  ".join(f"{n} Δ={C[e,di]:+.2f}" for di, n in enumerate(DOMAIN_NAMES)))

    # correlation between routing preference and causal impact across (e,d)
    rr, cc = R.ravel(), C.ravel()
    if rr.std() > 0 and cc.std() > 0:
        corr = float(np.corrcoef(rr, cc)[0, 1])
        print(f"\n  Pearson corr( P(e|d) , ΔPPL[e,d] ) = {corr:.3f}  "
              f"({'functional specialization' if corr > 0.3 else 'weak/none'})")

    print("\n=== Shared-expert ablation: ΔPPL per domain ===")
    for n in DOMAIN_NAMES:
        ppl = domain_ppl(model, eval_sets[n], device, args.batch, ablate_shared=True)
        print(f"  {n:4s}: baseline={base[n]:.2f} -> ablated={ppl:.2f}  Δ={ppl-base[n]:+.2f}")

    if args.out:
        np.savez(args.out, nmi_layer=np.array(nmi_layer), R=R, C=C,
                 baseline=np.array([base[n] for n in DOMAIN_NAMES]))
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
