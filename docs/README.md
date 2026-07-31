# Study: expert specialization and the shared-expert gradient race in small MoEs

This directory holds a self-contained empirical study built on the Tiny-MoE codebase. All
experiments run on a single GPU; the analysis toolkit is in `../analysis/`.

## Documents

| File | Contents |
|---|---|
| **[paper.md](paper.md)** | The workshop-length write-up: abstract, methods, results (F8+F9 core), discussion, limitations. **Start here.** |
| [appendix.md](appendix.md) | Metric definitions/formulas, hyperparameters, full result tables (F2/F3/F4/F8/F9/F10 + retractions), exact reproduction commands. |
| [results.md](results.md) | The running results log: all 10 findings with numbers, in the order discovered (includes the two self-corrections). |
| [experiment-protocol.md](experiment-protocol.md) | The original experimental protocol (design, controls, compute plan). |
| [references.bib](references.bib) | Citations. |
| [figures/](figures/) | Publication figures (`make_figures.py` regenerates them from the logged numbers). |

## The one-paragraph result

In small shared-expert MoEs, the routed experts are not redundant — they are out-competed by the
shared expert in a **gradient-exposure race** (the shared expert is updated on 100% of tokens vs.
~k/E per routed expert). This race resolves with training: branch importance is balanced by
~295M tokens (Figure 1). It is **causally controllable** — slowing the shared expert's learning
rate shifts the routed-activation crossover earlier, dose-dependently, at no perplexity cost
(Figure 2). Practical upshot: do not diagnose expert redundancy on an undertrained MoE, and to
engage the routed experts sooner, slow the shared expert.

## Core findings

- **F8** (observation): routed "redundancy" is an undertraining artifact; crossover ~295M tokens (4 seeds).
- **F9** (mechanism + control): a gradient-exposure race, controllable by the shared-expert LR at zero PPL cost.
- **F10** (scope): the #experts dependence is non-monotonic; the clean ∝E/k law is refuted.

Supporting (F2–F4): specialization is by surface form > domain > frequency, not position;
load-balancing controls specialization; the shared expert enables specialization when balancing
is off. Honest negatives (F7, distillation) and two retracted early findings are reported in full.

## Reproducing

Environment: a venv with `torch==2.10.0+cu128`, `datasets==3.6.0`, `transformers`, `safetensors`,
`numpy`, `sentencepiece`. See `appendix.md` §D for the exact per-finding commands. Two repository
bug-fixes made during the study (absorbed-MLA causal mask; a broken CUDA `torch.dot`) are in the
git history.
