# Raw run record: what sets the shared/routed branch split

Experiments run **2026-08-02 → 2026-08-05** on a single RTX A6000 (`cluster5`), after the
version of `paper.md` currently in this repository was written. **These results supersede
parts of §5 and all of §6 of that paper — see "What this changes" below. Nothing here has
been integrated into `paper.md` / `results.md` yet.**

Raw logs live on the training host at `~/tiny_moe_study/{followup,capacity,collapse,collapse_s1,ratio}.log`,
alongside the driver scripts `run_{followup,capacity,collapse,collapse_s1,ratio}.sh`.
Checkpoints are under `~/tiny_moe_study/<run>/step_*`.

## Common setup

All runs below: 8 transformer layers unless noted, top-k 2, batch 64, sequence length 512,
lr 3e-4, warmup 1500, aux-coef 1e-2, 30000 steps ≈ 983M tokens, checkpoint every 3000 steps.
Analysis is `analysis.race` with `--tokens-per-domain 60000 --skip 3000`.
`I` denotes the per-expert (routed) intermediate size, `I_shared` the shared expert's width.
The shared branch is a single expert active on every token; each routed expert sees ~k/E of tokens.
"Active routed capacity" means k·I, the routed width a single token actually flows through.

The reported quantity is `shared/(sh+rt)` — the shared branch's share of total ablation impact.
`±` is the spread across seeds; `±0.00` means a single seed, **not** zero variance.

## Summary

Minimum share reached over the trajectory, and share at 983M tokens.

| run | layers | E | I | I_shared | ratio I/I_shared | active k·I | params | seeds | min share | @983M | final PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `e8i2048` | 8 | 8 | 2048 | 1024 | 2.0 | 4096 | 234.4M | 2 | 0.37 | 0.40 | 2.95 |
| `e4wide` | 8 | 4 | 2048 | 1024 | 2.0 | 4096 | 133.7M | 2 | **0.25** | 0.28 | 2.99 |
| `sh512` | 8 | 8 | 1024 | 512 | 2.0 | 2048 | 127.4M | 2 | 0.28 | 0.34 | 3.01 |
| `base1b` (ref) | 8 | 8 | 1024 | 1024 | 1.0 | 2048 | 133.7M | 4 | 0.47 | 0.50 | 3.00 |
| `e16cap` | 8 | 16 | 512 | 1024 | 0.5 | 1024 | 133.7M | 2 | 0.63 | 0.67 | 3.00 |
| `e32cap` | 8 | 32 | 256 | 1024 | 0.25 | 512 | 133.8M | 2 | 0.72 | 0.75 | 3.02 |
| `e8i256` | 8 | 8 | 256 | 1024 | 0.25 | 512 | 58.2M | 2 | 0.79 | 0.80 | 3.12 |
| `sh4096` | 8 | 8 | 1024 | 4096 | 0.25 | 2048 | 171.5M | 2 | **0.81** | 0.82 | 2.97 |
| `scale16_1b` (ref) | 16 | 8 | 1024 | 1024 | 1.0 | 2048 | 251.0M | 1 | 0.54 | 0.59 | 2.90 |
| `dyn16` | 16 | 8 | 1024 | 1024 | 1.0 | 2048 | 251.0M | 1 | 0.14 | 0.21 | 2.89 |

`base1b` and `scale16_1b` are the already-published baselines (`finding8.log`, `extend.log`),
repeated here as reference rows. `dyn16` is the dynamic shared-LR run (see below), not an
architecture variant.

Perplexities are **not** comparable across rows with different parameter counts
(58.2M → 234.4M). Only the share is.

### Grouped by width ratio

| ratio I/I_shared | 2.0 | 1.0 | 0.5 | 0.25 |
|---|---|---|---|---|
| min share observed | 0.25 / 0.28 / 0.37 | 0.47 / 0.53 / 0.55 / 0.56 | 0.63 | 0.72 / 0.79 / 0.81 |

The ratio-1.0 entries include the four ratio-1.0 points of the published E-sweep
(E ∈ {4,8,16,32} at I = I_shared = 1024; see `results.md` Finding 10). The bands do not
overlap: the ratio orders every configuration measured, across 8× in ratio, 4× in parameter
count and 8× in expert count. Within a ratio level the residual spread is ~0.1 and is **not**
consistently explained by E, by absolute capacity, or by total routed parameters — at ratio 2.0
more experts raises the share (E=4 → 0.25 vs E=8 → 0.37) while at ratio 0.25 more experts
lowers it (E=8 → 0.79 vs E=32 → 0.72).

### The two disentangling contrasts

Both hold E=8 and k=2 fixed, so expert count and routing are constant throughout.

**Varying routed width** (`e8i256` / `base1b` / `e8i2048`, I_shared pinned at 1024):

| I | 256 | 1024 | 2048 |
|---|---|---|---|
| min share | 0.79 | 0.47 | 0.37 |

**Varying shared width** (`sh4096` / `base1b` / `sh512`, I pinned at 1024 — routed active
capacity is *identical* in all three, 2048):

| I_shared | 4096 | 1024 | 512 |
|---|---|---|---|
| min share | 0.81 | 0.47 | 0.28 |

The second contrast is the decisive one: the routed branch is byte-for-byte the same in all
three runs, and the split still moves by 0.53. Absolute routed capacity therefore does not
govern the split. The two contrasts agree wherever they meet the same ratio (0.79 vs 0.81 at
ratio 0.25; 0.37 vs 0.28 at ratio 2.0) despite parameter counts differing by up to 4×.

## What this changes

1. **Finding 10 as published is wrong on two counts.** Its non-monotonic #experts curve
   (0.56 / 0.48 / 0.53 / 0.55 for E = 4/8/16/32) held per-expert width fixed, so total routed
   capacity grew with E and confounded the comparison; and its reported perplexity gain with E
   (3.28 → 2.90) was a parameter-count effect — at matched parameters the perplexity is flat
   (2.99 / 3.00 / 3.00 / 3.02 for E = 4/8/16/32 in `e4wide`/`base1b`/`e16cap`/`e32cap`).
   Expert count is not what governs the split; the branch width ratio is.
2. **The ∝E/k prediction of Finding 9 is not supported** as the determinant of where the split
   lands. The gradient-exposure race remains causally demonstrated by the shared-LR control;
   what these runs refute is the extrapolation from that mechanism to a law over E.
3. **§5's caveat that the 16-layer model never crosses to balance within 1B tokens is removed**
   by the `dyn16` result below.

## Dynamic shared-LR (`dyn16`) — hysteresis

16 layers, 251.0M params, shared-expert LR multiplier ramped from ×0.1 up to ×1.0 by step 9000
(~295M tokens), 1 seed. Compare to the published 16-layer runs: baseline ×1.0 and static ×0.1.

| tokens | baseline ×1.0 | static ×0.1 | ramp 0.1→1.0 |
|---|---|---|---|
| 98M | 0.84 | 0.21 | 0.42 |
| 295M (ramp ends) | 0.58 | 0.14 | 0.15 |
| 491M | 0.54 | 0.16 | 0.16 |
| 983M | 0.59 | — | 0.21 |
| final PPL | 2.90 | — | 2.89 |

Once the handicap is lifted the reallocation does not undo: the share stays at 0.15–0.21
through 983M tokens instead of returning toward the baseline's 0.59, at unchanged perplexity.
The shared branch does slowly re-assert (0.14 → 0.21 between 295M and 983M), far more slowly
than the baseline's rise. **Single seed, single ramp endpoint, single depth** — this needs
replication and a dose-response before it can carry weight.

## Raw tables

Columns: `step | basePPL | Δrm shared | Δrm routed | routed NMI | shared/(sh+rt)`.

### D1 — `e4wide`, E=4 @ I=2048, I_shared=1024, 2 seeds (`followup.log`)

```
     0 |   10.43 |   +0.49±0.01 |   +0.00±0.00 |     0.0004 |          1.00
  3000 |    4.17 |   +2.76±0.18 |   +1.07±0.02 |     0.0112 |          0.72
  6000 |    3.62 |   +1.54±0.03 |   +2.19±0.02 |     0.0172 |          0.41
  9000 |    3.40 |   +1.21±0.03 |   +2.69±0.01 |     0.0187 |          0.31
 12000 |    3.28 |   +1.12±0.04 |   +3.06±0.06 |     0.0194 |          0.27
 15000 |    3.19 |   +1.11±0.05 |   +3.24±0.06 |     0.0199 |          0.26
 18000 |    3.12 |   +1.13±0.04 |   +3.32±0.04 |     0.0209 |          0.25
 21000 |    3.06 |   +1.20±0.04 |   +3.42±0.03 |     0.0211 |          0.26
 24000 |    3.02 |   +1.27±0.03 |   +3.47±0.03 |     0.0210 |          0.27
 27000 |    2.99 |   +1.32±0.04 |   +3.47±0.02 |     0.0207 |          0.28
 30000 |    2.99 |   +1.33±0.04 |   +3.47±0.02 |     0.0208 |          0.28
```

### D2 — `dyn16`, 16 layers, shared-LR ramp 0.1→1.0 by step 9000, 1 seed (`followup.log`)

```
     0 |   10.41 |   +0.50±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.12 |   +1.58±0.00 |   +2.18±0.00 |     0.0127 |          0.42
  6000 |    3.56 |   +0.95±0.00 |   +3.10±0.00 |     0.0143 |          0.23
  9000 |    3.33 |   +0.63±0.00 |   +3.50±0.00 |     0.0154 |          0.15
 12000 |    3.20 |   +0.58±0.00 |   +3.58±0.00 |     0.0147 |          0.14
 15000 |    3.11 |   +0.68±0.00 |   +3.57±0.00 |     0.0145 |          0.16
 18000 |    3.03 |   +0.74±0.00 |   +3.53±0.00 |     0.0144 |          0.17
 21000 |    2.96 |   +0.83±0.00 |   +3.63±0.00 |     0.0144 |          0.19
 24000 |    2.92 |   +0.92±0.00 |   +3.57±0.00 |     0.0144 |          0.20
 27000 |    2.89 |   +0.99±0.00 |   +3.63±0.00 |     0.0146 |          0.21
 30000 |    2.89 |   +1.00±0.00 |   +3.67±0.00 |     0.0145 |          0.21
```

### `e16cap`, E=16 @ I=512, I_shared=1024, 2 seeds (`capacity.log`)

```
     0 |   10.42 |   +0.50±0.01 |   +0.00±0.00 |     0.0017 |          1.00
  3000 |    4.15 |   +3.67±0.01 |   +0.79±0.01 |     0.0163 |          0.82
  6000 |    3.62 |   +3.15±0.10 |   +1.30±0.02 |     0.0173 |          0.71
  9000 |    3.40 |   +2.89±0.08 |   +1.61±0.08 |     0.0188 |          0.64
 12000 |    3.28 |   +2.92±0.04 |   +1.72±0.04 |     0.0184 |          0.63
 15000 |    3.20 |   +3.05±0.04 |   +1.80±0.09 |     0.0184 |          0.63
 18000 |    3.13 |   +3.21±0.03 |   +1.80±0.06 |     0.0191 |          0.64
 21000 |    3.07 |   +3.45±0.05 |   +1.84±0.06 |     0.0186 |          0.65
 24000 |    3.03 |   +3.65±0.07 |   +1.86±0.06 |     0.0187 |          0.66
 27000 |    3.01 |   +3.77±0.06 |   +1.89±0.09 |     0.0187 |          0.67
 30000 |    3.00 |   +3.79±0.07 |   +1.90±0.09 |     0.0188 |          0.67
```

### `e32cap`, E=32 @ I=256, I_shared=1024, 2 seeds (`capacity.log`)

```
     0 |   10.42 |   +0.48±0.01 |   -0.00±0.00 |     0.0032 |          1.00
  3000 |    4.15 |   +3.63±0.26 |   +0.74±0.02 |     0.0226 |          0.83
  6000 |    3.62 |   +3.48±0.22 |   +1.16±0.06 |     0.0237 |          0.75
  9000 |    3.41 |   +3.48±0.07 |   +1.33±0.03 |     0.0229 |          0.72
 12000 |    3.30 |   +3.63±0.15 |   +1.40±0.03 |     0.0237 |          0.72
 15000 |    3.22 |   +3.84±0.17 |   +1.48±0.01 |     0.0227 |          0.72
 18000 |    3.15 |   +4.08±0.25 |   +1.50±0.03 |     0.0226 |          0.73
 21000 |    3.09 |   +4.40±0.24 |   +1.52±0.01 |     0.0228 |          0.74
 24000 |    3.05 |   +4.64±0.27 |   +1.54±0.01 |     0.0226 |          0.75
 27000 |    3.03 |   +4.78±0.29 |   +1.57±0.00 |     0.0225 |          0.75
 30000 |    3.02 |   +4.81±0.29 |   +1.58±0.00 |     0.0225 |          0.75
```

### `e8i256`, E=8 @ I=256, I_shared=1024, 2 seeds (`collapse_s1.log`)

```
     0 |   10.42 |   +0.50±0.01 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.23 |   +4.70±0.17 |   +0.44±0.00 |     0.0103 |          0.91
  6000 |    3.72 |   +4.38±0.04 |   +0.79±0.01 |     0.0134 |          0.85
  9000 |    3.51 |   +4.30±0.20 |   +0.98±0.01 |     0.0140 |          0.81
 12000 |    3.39 |   +4.39±0.28 |   +1.11±0.01 |     0.0139 |          0.80
 15000 |    3.31 |   +4.55±0.31 |   +1.19±0.01 |     0.0133 |          0.79
 18000 |    3.24 |   +4.82±0.34 |   +1.25±0.02 |     0.0136 |          0.79
 21000 |    3.19 |   +5.10±0.34 |   +1.30±0.01 |     0.0137 |          0.80
 24000 |    3.15 |   +5.32±0.37 |   +1.33±0.03 |     0.0132 |          0.80
 27000 |    3.13 |   +5.46±0.38 |   +1.38±0.01 |     0.0130 |          0.80
 30000 |    3.12 |   +5.48±0.38 |   +1.39±0.02 |     0.0130 |          0.80
```

### `e8i2048`, E=8 @ I=2048, I_shared=1024, 2 seeds (`collapse_s1.log`)

```
     0 |   10.42 |   +0.51±0.01 |   -0.00±0.00 |     0.0009 |          1.00
  3000 |    4.15 |   +2.87±0.05 |   +1.06±0.05 |     0.0146 |          0.73
  6000 |    3.62 |   +2.01±0.04 |   +2.00±0.00 |     0.0181 |          0.50
  9000 |    3.38 |   +1.66±0.11 |   +2.37±0.02 |     0.0202 |          0.41
 12000 |    3.25 |   +1.55±0.08 |   +2.62±0.07 |     0.0225 |          0.37
 15000 |    3.16 |   +1.55±0.07 |   +2.64±0.06 |     0.0226 |          0.37
 18000 |    3.08 |   +1.62±0.06 |   +2.73±0.13 |     0.0233 |          0.37
 21000 |    3.02 |   +1.76±0.06 |   +2.79±0.10 |     0.0238 |          0.39
 24000 |    2.98 |   +1.85±0.08 |   +2.79±0.10 |     0.0243 |          0.40
 27000 |    2.95 |   +1.93±0.07 |   +2.86±0.14 |     0.0240 |          0.40
 30000 |    2.95 |   +1.95±0.07 |   +2.86±0.14 |     0.0239 |          0.40
```

Seed 0 alone gave min share 0.81 and 0.35 for these two configurations; adding seed 1 moved
them by 0.02. Seed noise on this quantity is therefore ~0.02, well below the effects reported
above.

### `sh512`, E=8 @ I=1024, I_shared=512, 2 seeds (`ratio.log`)

```
     0 |   10.49 |   +0.43±0.00 |   +0.00±0.00 |     0.0008 |          1.00
  3000 |    4.15 |   +1.96±0.12 |   +1.66±0.06 |     0.0157 |          0.54
  6000 |    3.61 |   +1.41±0.11 |   +2.64±0.11 |     0.0201 |          0.35
  9000 |    3.41 |   +1.27±0.11 |   +2.94±0.07 |     0.0231 |          0.30
 12000 |    3.29 |   +1.20±0.14 |   +3.14±0.12 |     0.0231 |          0.28
 15000 |    3.20 |   +1.21±0.15 |   +3.12±0.08 |     0.0220 |          0.28
 18000 |    3.13 |   +1.32±0.16 |   +3.11±0.03 |     0.0231 |          0.30
 21000 |    3.07 |   +1.41±0.18 |   +3.20±0.09 |     0.0231 |          0.31
 24000 |    3.03 |   +1.52±0.19 |   +3.18±0.06 |     0.0229 |          0.32
 27000 |    3.01 |   +1.59±0.19 |   +3.20±0.03 |     0.0227 |          0.33
 30000 |    3.01 |   +1.61±0.20 |   +3.19±0.03 |     0.0226 |          0.34
```

### `sh4096`, E=8 @ I=1024, I_shared=4096, 2 seeds (`ratio.log`)

```
     0 |   10.42 |   +0.51±0.02 |   -0.00±0.00 |     0.0009 |          1.00
  3000 |    4.16 |   +5.31±0.02 |   +0.25±0.01 |     0.0080 |          0.95
  6000 |    3.68 |   +4.89±0.06 |   +0.56±0.01 |     0.0099 |          0.90
  9000 |    3.41 |   +4.63±0.23 |   +0.76±0.02 |     0.0098 |          0.86
 12000 |    3.27 |   +4.44±0.25 |   +0.88±0.03 |     0.0099 |          0.83
 15000 |    3.18 |   +4.51±0.17 |   +0.97±0.02 |     0.0104 |          0.82
 18000 |    3.10 |   +4.70±0.21 |   +1.04±0.04 |     0.0101 |          0.82
 21000 |    3.04 |   +4.93±0.16 |   +1.12±0.02 |     0.0104 |          0.81
 24000 |    2.99 |   +5.12±0.15 |   +1.14±0.04 |     0.0102 |          0.82
 27000 |    2.97 |   +5.26±0.12 |   +1.18±0.03 |     0.0103 |          0.82
 30000 |    2.97 |   +5.29±0.12 |   +1.18±0.03 |     0.0103 |          0.82
```

Seed 1 of `sh512` and `sh4096` finished after the first version of this record was written;
the tables above are the pooled two-seed analyses. Adding seed 1 moved the minimum share by
0.03 (`sh512`, 0.31 → 0.28) and 0.01 (`sh4096`, 0.82 → 0.81) — consistent with the ~0.02 seed
noise measured on `e8i256` / `e8i2048`, and it widens the shared-width contrast slightly rather
than narrowing it.

## Reproduction

```bash
# D1 (e4wide) — routed capacity matched to the E=8 baseline at equal total parameters
python -m analysis.pilot_train --num-layers 8 --num-experts 4 --top-k 2 \
  --moe-intermediate 2048 --shared-intermediate 1024 --steps 30000 --batch 64 --seq-len 512 \
  --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s --out e4wide_s$s

# D2 (dyn16) — shared-LR ramped back to parity by step 9000
python -m analysis.pilot_train --num-layers 16 --num-experts 8 --top-k 2 --steps 30000 \
  --batch 64 --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 \
  --shared-lr-mult 0.1 --shared-ramp-end 9000 --seed 0 --out dyn16_s0

# capacity-matched #experts sweep: E*I held at 8192
python -m analysis.pilot_train --num-layers 8 --num-experts $E --top-k 2 \
  --moe-intermediate $((8192/E)) --shared-intermediate 1024 --steps 30000 --batch 64 \
  --seq-len 512 --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s \
  --out e${E}cap_s$s

# routed-width contrast at E=8 (I in 256, 1024, 2048)
python -m analysis.pilot_train --num-layers 8 --num-experts 8 --top-k 2 \
  --moe-intermediate $I --shared-intermediate 1024 --steps 30000 --batch 64 --seq-len 512 \
  --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s --out e8i${I}_s$s

# shared-width contrast at E=8 (I_shared in 512, 1024, 4096), routed branch identical
python -m analysis.pilot_train --num-layers 8 --num-experts 8 --top-k 2 \
  --moe-intermediate 1024 --shared-intermediate $IS --steps 30000 --batch 64 --seq-len 512 \
  --lr 3e-4 --warmup 1500 --aux-coef 1e-2 --save-every 3000 --seed $s --out sh${IS}_s$s

# analysis (pool seeds with a comma-separated list)
python -m analysis.race --runs <run_dir>[,<run_dir>...] --tokens-per-domain 60000 --skip 3000
```
