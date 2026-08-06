# Study: expert specialization and the shared-expert gradient race in small MoEs

This directory holds a self-contained empirical study built on the Tiny-MoE codebase. All
experiments run on a single GPU; the analysis toolkit is in `../analysis/`.

## Documents

| File | Contents |
|---|---|
| **[paper.md](paper.md)** | The workshop-length write-up: abstract, methods, results (F8+F9+F11 core), discussion, limitations. **Start here.** |
| [appendix.md](appendix.md) | Metric definitions/formulas, hyperparameters, full result tables (F2/F3/F4/F8/F9/F11 + retractions), exact reproduction commands. |
| [results.md](results.md) | The running results log: all 12 findings with numbers, in the order discovered (includes the three self-corrections). |
| [branch-split-runs.md](branch-split-runs.md) | Raw per-checkpoint record of the width-ratio run series (F11/F12), with provenance and reproduction commands. |
| [experiment-protocol.md](experiment-protocol.md) | The original experimental protocol (design, controls, compute plan). |
| [references.bib](references.bib) | Citations. |
| [figures/](figures/) | Publication figures (`make_figures.py` regenerates them from the logged numbers). |

## The one-paragraph result

In small shared-expert MoEs, the routed experts are not redundant — they are out-competed by the
shared expert in a **gradient-exposure race** (the shared expert is updated on 100% of tokens vs.
~k/E per routed expert). This race resolves with training: branch importance is balanced by
~295M tokens (Figure 1). It is **causally controllable** — slowing the shared expert's learning
rate shifts the routed-activation crossover earlier, dose-dependently, at no perplexity cost
(Figure 2), and a *temporary* handicap is enough — the reallocation persists once it is lifted.
Where the split finally lands is set by the **routed:shared width ratio**, not by the number of
experts (Figure 3). Practical upshot: do not diagnose expert redundancy on an undertrained MoE;
to engage the routed experts sooner, slow the shared expert; to engage them more, do not make the
shared expert wide.

## Core findings

- **F8** (observation): routed "redundancy" is an undertraining artifact; crossover ~295M tokens (4 seeds).
- **F9** (mechanism + control): a gradient-exposure race, controllable by the shared-expert LR at zero PPL cost.
- **F11** (what sets the split): the routed:shared width ratio orders every configuration measured
  (0.25 → 0.81 shared share over 8× in ratio); expert count essentially does not matter. Supersedes F10.
- **F12** (hysteresis): a temporary shared-LR handicap permanently reallocates computation, at unchanged PPL.

Supporting (F2–F4): specialization is by surface form > domain > frequency, not position;
load-balancing controls specialization; the shared expert enables specialization when balancing
is off. Honest negatives (F7, distillation) and three retracted findings — including F10's
non-monotonic #experts law, confounded by total routed capacity — are reported in full.

## Reproducing

Environment: a venv with `torch==2.10.0+cu128`, `datasets==3.6.0`, `transformers`, `safetensors`,
`numpy`, `sentencepiece`. See `appendix.md` §D for the exact per-finding commands. Two repository
bug-fixes made during the study (absorbed-MLA causal mask; a broken CUDA `torch.dot`) are in the
git history.
