# Expert Specialization in a Small Multi-Domain MoE — Experimental Protocol

**Status:** draft v1 · **Hardware:** single NVIDIA RTX A6000 (48 GB) · **Codebase:** Tiny-MoE

## 1. Research question

> In a small Mixture-of-Experts language model trained on a *controlled, domain-labelled*
> corpus, do experts specialize, along which axes, when does specialization emerge during
> training, and is it *functional* (causally load-bearing) rather than epiphenomenal?

The distinctive lever of this codebase: the pre-training mixture has **three sources with
known labels** — FineWeb-Edu (web/edu), Cosmopedia v2 (synthetic educational),
OpenWebMath (mathematics) — **and** a **shared-expert + routed-expert** architecture. Unlike
large-scale studies over an undifferentiated mix, every evaluation token here carries a
ground-truth domain label, which makes routing attributable.

### Sub-questions (each yields a figure)

- **Q1 — Shared/routed division of labour.** Does the shared expert absorb "common"
  computation while routed experts specialize? (DeepSeekMoE's motivating claim, rarely
  tested in a controlled small-scale setting.)
- **Q2 — Axis of specialization.** Semantic *domain* vs mere *token identity* (math text has
  more digits/symbols — a routing pattern could reflect token frequency, not domain meaning).
- **Q3 — Emergence.** When during training does routing crystallize — before or after the
  loss plateau?
- **Q4 — Drivers.** Effect of the shared expert, the load-balancing coefficient, and top-k on
  the amount and shape of specialization.

## 2. Three methodological legs

### A. Observational — routing structure

For each layer ℓ, over a domain-balanced held-out set, record the top-k expert assignment of
every token together with its domain label. Compute:

- **Routing matrix** `R[ℓ][e, d] = P(expert = e | domain = d)`.
- **Specialization score** `NMI(ℓ) = I(Expert ; Domain) / H(Domain)` ∈ [0, 1], per layer.
- **Expert purity** `H(Domain | expert = e)` (low ⇒ domain-specialized expert).
- **Shared vs routed magnitude** `‖routed_out‖ / ‖shared_out‖` per token, per domain (Q1).
- **Token-identity control (Q2):** routing consistency of the *same* token type across
  domains. If a token routes identically everywhere → token-driven; if it routes differently
  by domain → context/semantic-driven. Reported as per-token routing-distribution divergence
  across domains, aggregated over the shared vocabulary.

### B. Causal / interventional — functional specialization

- **Expert ablation → per-domain ΔPPL.** Mask expert `e` (router logit → −∞), measure the
  perplexity increase on *each* domain separately. Build a causal impact matrix `C[e, d]` and
  **correlate it with the observational routing matrix** `R`. High correlation ⇒ observed
  routing is functionally meaningful.
- **Shared-expert ablation → ΔPPL** by domain and by token-frequency bucket (direct test of Q1).

### C. Longitudinal — emergence (nearly free; reuses checkpoints)

Replay the leg-A metrics on every saved checkpoint → curve of `NMI(ℓ)` vs training step,
overlaid on the training/val loss curve (Q3).

## 3. Controlled variable manipulation (leg D — training variants)

Vary **one knob at a time** (all already exposed in `training/training_configs.py`), holding
tokens-seen and all else fixed. Priority order for a single-A6000 budget:

| Variant | Knob | Values | Tests |
|---|---|---|---|
| **D1 — shared expert** | remove shared expert (`num_shared = 0`) | {with, without} | Q1 causally; does the shared expert *increase* routed specialization? |
| **D2 — load balancing** | `router_aux_loss_coef` | {0, 1e-2 (default), 1e-1} | Balance ↔ specialization tension |
| D3 — granularity *(optional)* | `num_experts` at matched active params | {4, 8, 16} | More experts ⇒ sharper specialization? |
| D4 — routing hardness *(optional)* | `num_experts_per_token` | {1, 2} | Harder routing ⇒ sharper specialization? |

D1 and D2 are the headline variants; D3/D4 only if compute allows. Note: the current model has
no `num_shared` switch — D1 requires a one-line config/arch flag (zero out the shared branch).

## 4. Confound controls (anticipating reviewer objections)

| Risk | Control |
|---|---|
| "MI is spurious" | **Untrained-model floor** + **label-permutation control** (shuffle domain labels ⇒ NMI must collapse to ≈ 0). |
| Domain token-frequency differences | Report metrics raw **and** token-conditioned / on a shared-vocabulary subset. |
| Capacity dropping distorts routing | Analyse in **inference mode** (no capacity limit), not the training forward. |
| Seed noise | ≥ 2 seeds for the reference model. |
| Misleading aggregation | Always report **per layer**, never only the layer-mean. |

## 5. Compute plan (single A6000)

The A6000 ≈ the original 2×T4 throughput; a full 30k-step run ≈ 2–4 days. Staged:

- **Pilot (≈ ½ day).** Reference at ~2–3k steps + the *entire* analysis pipeline end-to-end.
  Goal: confirm NMI exceeds the permutation control **before** committing to long runs.
- **Tier 0 (required).** 1 reference, ~20k steps (~2.6 B tokens), 1 seed + full analysis
  (A + B + C). Already a workshop-grade result.
- **Tier 1.** + 1 reference seed (robustness) + D1 (no-shared variant).
- **Tier 2.** + D2 sweep {0, 1e-1}.

≈ 5 trainings, sequential over a few weeks. Variants may be **shortened** — they establish a
trend, not a SOTA number.

## 6. Pilot trainer (this repo)

Because the pilot needs neither DeepSpeed, 8-bit optimizers, nor multi-GPU, it uses a
**standalone single-GPU PyTorch loop** (`analysis/pilot_train.py`) that reuses the repository
model (`training.model_train.Transformer`) and streaming data (`training.data`). Only
dependency beyond torch: `datasets`. The production `training/trainer.py` (Accelerate +
DeepSpeed) remains the path for full Tier-0+ runs on multi-GPU hardware.

## 7. Instrumentation (this repo, `analysis/`)

1. `analysis/domain_eval.py` — per-domain held-out streams, **no cross-domain packing**, one
   domain label per sequence.
2. `analysis/model_analysis.py` — analysis model whose MoE returns per-token routing
   (`topk_indices`, gate weights) and shared/routed norms; supports **expert ablation** and
   **shared-expert ablation**.
3. `analysis/metrics.py` — routing matrix, NMI, purity, token-identity control, permutation
   control.
4. `analysis/run_analysis.py` — load a checkpoint → per-domain eval → dump routing tensors →
   metrics + ablation ΔPPL matrix.
5. `analysis/pilot_train.py` — standalone single-GPU pilot trainer.

## 8. Deliverables / expected contributions

1. A reusable **methodology** for measuring expert specialization with a known multi-domain
   corpus, combining observational + causal + longitudinal analysis.
2. Empirical findings on **what** experts specialize by (domain vs token), **when** it emerges,
   and the **shared/routed** division of labour.
3. A **causal** expert×domain impact matrix validating that observed routing is load-bearing.

**Target venue:** interpretability workshop (e.g. BlackboxNLP @ EMNLP) or a short paper — a
controlled empirical study, explicitly at small scale.
