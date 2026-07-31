# Appendix

Companion to `paper.md`. Full tables, metric definitions, hyperparameters, and exact
reproduction commands. All experiments were run on a single NVIDIA RTX A6000.

## A. Metric definitions

Let `E` be the number of routed experts, `k` the top-k, and let each analysed token carry a
label from a discrete axis (domain `d ∈ {web, edu, math}`, surface class, frequency bucket, or
sequence-position bucket). Routing is captured in inference mode (no capacity dropping).

**Whole-branch ablation (causal importance).** For a branch *b* (all routed experts, or the
shared expert), `Δ_b = PPL(model with b zeroed) − PPL(model)`, measured on held-out per-domain
text and averaged over domains. We use whole-branch ablation because single-expert ablation is
confounded by top-2 rerouting to the surviving experts. The **shared share** is
`Δ_shared / (Δ_shared + Δ_routed)`; 1.0 means the shared branch carries all ablation impact,
0.5 means the branches are balanced.

**Soft NMI.** With top-k indices and gate weights per token, the joint count
`J[e, a] = Σ_t Σ_{j} g_{t,j} · 1[topk_idx_{t,j}=e] · 1[axis_t=a]`; then
`NMI = I(E;A)/H(A)` from `J`. Averaged over layers. A label-permutation null (shuffle `a`)
is ≈ 0 in every layer.

**Conditional NMI.** `NMI(E; A | C) = I(E;A|C)/H(A|C) = Σ_c p(c) I(E;A|C=c) / Σ_c p(c) H(A|C=c)`,
computed on tokens whose conditioner value `c` has ≥ `min_count` occurrences. Used to test
whether one specialization axis explains another.

**Load imbalance.** Mean over layers of the standard deviation of the top-1 expert load
distribution `P(e)`. 0 = perfectly balanced.

**Permutation test.** Exact two-sided test on the difference of means between two seed groups;
with `n` seeds per group the minimum achievable two-sided p is `2 / C(2n, n)` (0.10 for 3v3,
0.029 for 4v4).

## B. Model and training hyperparameters

| Component | Value |
|---|---|
| Parameters | 133.7M (E=8); 83.4M (E=4); larger for E=16/32 |
| Layers / hidden / heads | 8 / 512 / 8 |
| Attention | MLA (kv_lora_rank 96, qk_nope 48, qk_rope 16), SDPA |
| Routed experts / top-k / shared | 8 / 2 / 1 |
| MoE intermediate size | 1024 |
| Capacity factor (train) | 1.25 |
| Vocab / context | 32000 (Mistral) / 512 |
| Optimizer | AdamW, betas (0.9, 0.95), wd 0.1, grad clip 1.0 |
| LR / schedule | 3e-4, cosine to 0, warmup 5% |
| Precision | bf16 autocast |
| aux-loss / z-loss coef | 1e-2 / 1e-3 (unless swept) |
| Data mix | FineWeb-Edu 60% / Cosmopedia-v2 25% / OpenWebMath 15% |

Token counts: `tokens = steps × batch × 512`. Batch 64 → step 3000 ≈ 98M, 6000 ≈ 197M,
9000 ≈ 295M, 15000 ≈ 491M, 30000 ≈ 983M tokens.

## C. Full result tables

### C.1 Finding 8 — branch importance vs tokens (baseline, 4 seeds, batch 64, mean ± std)

| step | tokens | PPL | Δ remove shared | Δ remove routed | routed NMI | shared share |
|---|---|---|---|---|---|---|
| 0 | 0 | 10.42 | +0.49 ± 0.01 | −0.00 | 0.0008 | 1.00 |
| 3000 | 98M | 4.18 | +3.44 ± 0.17 | +0.82 ± 0.03 | 0.0121 | 0.81 |
| 6000 | 197M | 3.63 | +2.37 ± 0.17 | +1.58 ± 0.06 | 0.0148 | 0.60 |
| 9000 | 295M | 3.41 | +1.99 ± 0.14 | +1.98 ± 0.06 | 0.0161 | 0.50 |
| 12000 | 393M | 3.28 | +1.93 ± 0.18 | +2.14 ± 0.03 | 0.0167 | 0.47 |
| 15000 | 491M | 3.20 | +1.99 ± 0.16 | +2.25 ± 0.01 | 0.0168 | 0.47 |
| 18000 | 590M | 3.13 | +2.07 ± 0.16 | +2.27 ± 0.03 | 0.0175 | 0.48 |
| 21000 | 688M | 3.07 | +2.21 ± 0.16 | +2.38 ± 0.03 | 0.0177 | 0.48 |
| 24000 | 786M | 3.03 | +2.35 ± 0.17 | +2.41 ± 0.07 | 0.0175 | 0.49 |
| 27000 | 885M | 3.01 | +2.44 ± 0.17 | +2.49 ± 0.10 | 0.0175 | 0.50 |
| 30000 | 983M | 3.00 | +2.45 ± 0.18 | +2.48 ± 0.11 | 0.0174 | 0.50 |

### C.2 Finding 9 — shared-expert LR sweep (2 seeds), values at 98M tokens

| shared LR mult | Δ remove routed | Δ remove shared | shared share | PPL |
|---|---|---|---|---|
| ×1.0 | +0.85 | +3.46 | 0.80 | 4.16 |
| ×0.3 | +1.94 | +2.48 | 0.56 | 4.10 |
| ×0.1 | +4.46 | +0.72 | 0.14 | 4.10 |

Trajectories (shared share by tokens): ×0.3 → 0.56 (98M), 0.49 (197M), 0.50 (295M), 0.52 (491M);
×0.1 → 0.14 (98M), 0.10 (197M), 0.09 (295M), 0.10 (491M). PPL matches baseline throughout.

### C.3 Finding 10 — #experts sweep (top-2, batch 64), minimum shared share and final PPL

| E | min shared share | reached at | crosses 0.5 | final PPL |
|---|---|---|---|---|
| 4 | 0.56 | ~295M | no | 3.28 |
| 8 | 0.48 | ~390M | yes | 3.00 |
| 16 | 0.53 | ~450M | no | 2.94 |
| 32 | 0.55 | ~390M | no | 2.90 |

### C.4 Finding 2 — aux-loss sweep (4 seeds), at 98M tokens

| aux coef | soft NMI (domain) | load imbalance | PPL | (pairwise permutation tests all p=0.029) |
|---|---|---|---|---|
| 0 | 0.0338 ± 0.0056 | 0.187 ± 0.008 | 4.49 | |
| 1e-2 | 0.0085 ± 0.0015 | 0.020 ± 0.001 | 4.47 | |
| 1e-1 | 0.0010 ± 0.0002 | 0.018 ± 0.001 | 4.47 | |

### C.5 Finding 3 — axis of specialization (aux=0 models, 4 seeds) and disentanglement

soft NMI(Expert; axis): surface **0.050 ± 0.006**, domain 0.034 ± 0.006, frequency 0.028 ± 0.003,
position 0.0003 (all controls ≈ 0.000).

Conditional MI (retention of routing information after conditioning): domain | surface = 100%,
domain | frequency = 100%, surface | domain = 105%, surface | frequency = 93%, frequency |
surface = 82%. The axes are separable; domain is not a surface-form artifact.

### C.6 Finding 4 — shared × aux interaction (4 seeds)

| condition | routed soft NMI | Δ remove routed | Δ remove shared | PPL |
|---|---|---|---|---|
| aux=1e-2, with shared | 0.0085 | +0.55 | +4.00 | 4.47 |
| aux=1e-2, no shared | 0.0093 | +8.01 | — | 4.52 |
| aux=0, with shared | 0.0338 | +0.46 | +4.81 | 4.49 |
| aux=0, no shared | 0.0150 | +6.50 | — | 4.60 |

At aux=1e-2 the shared/no-shared NMI difference is n.s. (p=0.543); at aux=0, with-shared NMI
(0.0338) > no-shared (0.0150), p=0.029.

### C.7 Retractions and preliminary

- **Retraction (a):** no-shared NMI vs baseline at 3 seeds 0.0100 vs 0.0078 (clean separation),
  at 4 seeds 0.0093 ± 0.0013 vs 0.0085 ± 0.0015 (p=0.543).
- **Retraction (b) / dropout:** per-token shared dropout p=0.3 drives shared share → 0.01
  (remove-shared +0.08, remove-routed +5.63) but base PPL 4.05 vs 4.02 (no gain) and NMI
  0.0085 vs 0.0072 (n.s.).
- **Distillation (single seed, 98M tokens):** vs batch/step-matched CE control — soft NMI
  0.0095 vs 0.0063, Δ remove routed +1.74 vs +1.03, Δ remove shared +4.40 vs +3.85, PPL 4.09 vs
  3.92.
- **Batch confound:** CE at 98M tokens, batch 24 vs 64 — soft NMI 0.0063 vs 0.0141, Δ remove
  routed +1.03 vs +0.84.

## D. Reproduction commands

Environment: an isolated venv (`~/tmvenv`) with `torch==2.10.0+cu128`, `datasets==3.6.0`,
`transformers`, `safetensors`, `numpy`, `sentencepiece`. Use `HF_HUB_DISABLE_XET=1` for the
Mistral teacher download.

```bash
# Finding 8 — token trajectory (4 seeds); seed 0 == the reference run
for s in 0 1 2 3; do
  python -m analysis.pilot_train --num-layers 8 --num-experts 8 --top-k 2 \
    --steps 30000 --batch 64 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 \
    --save-every 3000 --seed $s --out out/base1b_s$s
done
python -m analysis.race --runs out/base1b_s0,out/base1b_s1,out/base1b_s2,out/base1b_s3 \
  --tokens-per-domain 60000 --skip 3000

# Finding 9 — shared-expert LR sweep (mechanism)
for m in 0.3 0.1; do for s in 0 1; do
  python -m analysis.pilot_train --steps 15000 --batch 64 --save-every 3000 \
    --shared-lr-mult $m --seed $s --out out/shlr${m}_s$s
done; done   # x1.0 == the base1b runs
python -m analysis.race --runs out/shlr0.1_s0,out/shlr0.1_s1 --tokens-per-domain 60000 --skip 3000

# Finding 10 — #experts law
for E in 4 16 32; do for s in 0 1; do
  python -m analysis.pilot_train --num-experts $E --top-k 2 --steps 30000 --batch 64 \
    --save-every 3000 --seed $s --out out/e${E}_s$s
done; done
python -m analysis.race --runs out/e16_s0,out/e16_s1 --tokens-per-domain 60000 --skip 3000

# Finding 2 — aux-loss sweep
for a in 0.0 0.1; do for s in 0 1 2 3; do
  python -m analysis.pilot_train --aux-coef $a --steps 3000 --batch 32 --seed $s --out out/aux${a}_s$s
done; done
python -m analysis.d2_experiment --group aux0=out/aux0.0_s0/step_3000,... \
  --group aux1e-2=... --group aux1e-1=out/aux0.1_s0/step_3000,...

# Finding 3 — axis of specialization + disentanglement (on aux=0 models)
python -m analysis.multi_axis   --ckpts out/aux0.0_s0/step_3000,... --tokens-per-domain 80000
python -m analysis.disentangle  --ckpts out/aux0.0_s0/step_3000,... --tokens-per-domain 80000

# Single-checkpoint full report (observational + causal, any model)
python -m analysis.run_analysis --ckpt out/base1b_s0/step_30000/model.safetensors \
  --config out/base1b_s0/step_30000/config.json --ablate-layer 4

# Distillation (Mistral-7B teacher)
python -m analysis.distill_train --steps 8000 --batch 24 --kd-alpha 0.9 --kd-temp 2.0 \
  --out out/distill98m
```

## E. Implementation notes and fixes

Two bugs were found and fixed during the study:
- **Absorbed-MLA causal mask.** The weight-absorbed MLA inference path applied no causal mask
  during prefill, so query positions attended to future tokens (only when `absorb_weights=True`).
  Verified on GPU: max-abs-diff to the causal reference dropped from 1.9e-1 to 6e-8 after the fix.
- **Broken CUDA `torch.dot`.** The load-balancing aux-loss used `torch.dot(importance, load)`,
  which raised SIGFPE on one torch CUDA build (2.10.0+cu128, driver 575/CUDA 12.9); replaced with
  the identical `(importance * load).sum()`.

Analysis is run with capacity dropping disabled (inference-equivalent MoE), which we verified is
numerically identical to the training MoE when no tokens are dropped.
