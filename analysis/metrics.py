"""Specialization metrics computed from captured routing decisions.

Primary routing variable = top-1 expert (the argmax route), which gives one
categorical expert per token and a clean mutual-information estimate.
"""
from __future__ import annotations
import numpy as np


def _joint(expert, domain, E, D):
    j = np.zeros((E, D), dtype=np.float64)
    np.add.at(j, (expert, domain), 1.0)
    return j


def routing_matrix(expert, domain, E, D):
    """P(expert | domain): columns sum to 1. Shape [E, D]."""
    counts = _joint(expert, domain, E, D)
    col = counts.sum(axis=0, keepdims=True)
    return counts / np.clip(col, 1e-12, None)


def nmi(expert, domain, E, D):
    """Normalized MI I(E;D)/H(D) in [0,1]."""
    j = _joint(expert, domain, E, D)
    total = j.sum()
    if total == 0:
        return 0.0
    p = j / total
    pe = p.sum(axis=1, keepdims=True)
    pd = p.sum(axis=0, keepdims=True)
    mask = p > 0
    mi = np.sum(p[mask] * np.log(p[mask] / (pe @ np.ones_like(pd))[mask] / (np.ones_like(pe) @ pd)[mask]))
    pdv = pd.ravel()
    hd = -np.sum(pdv[pdv > 0] * np.log(pdv[pdv > 0]))
    return float(mi / hd) if hd > 0 else 0.0


def purity(expert, domain, E, D, domain_names=None):
    """Per-expert: dominant domain, its share, and entropy of P(domain|expert)."""
    counts = _joint(expert, domain, E, D)
    rows = counts.sum(axis=1)
    out = []
    for e in range(E):
        if rows[e] == 0:
            out.append(dict(expert=e, n=0, dominant=None, share=0.0, entropy=0.0))
            continue
        pd = counts[e] / rows[e]
        ent = -np.sum(pd[pd > 0] * np.log(pd[pd > 0]))
        dom = int(pd.argmax())
        out.append(dict(
            expert=e, n=int(rows[e]),
            dominant=(domain_names[dom] if domain_names else dom),
            share=float(pd[dom]), entropy=float(ent),
        ))
    return out


def permutation_control(expert, domain, E, D, n_perm=30, seed=0):
    """NMI under shuffled domain labels — the null floor. Returns (mean, std)."""
    rng = np.random.default_rng(seed)
    vals = []
    dom = np.asarray(domain)
    for _ in range(n_perm):
        vals.append(nmi(expert, rng.permutation(dom), E, D))
    return float(np.mean(vals)), float(np.std(vals))


def _nmi_from_joint(j):
    total = j.sum()
    if total == 0:
        return 0.0
    p = j / total
    pe = p.sum(axis=1, keepdims=True)
    pd = p.sum(axis=0, keepdims=True)
    mask = p > 0
    mi = np.sum(p[mask] * np.log(p[mask] / (pe @ np.ones_like(pd))[mask] / (np.ones_like(pe) @ pd)[mask]))
    pdv = pd.ravel()
    hd = -np.sum(pdv[pdv > 0] * np.log(pdv[pdv > 0]))
    return float(mi / hd) if hd > 0 else 0.0


def _mi_h(j):
    """Return (MI in nats, H(target) in nats) for a joint count matrix [E, T]."""
    total = j.sum()
    if total == 0:
        return 0.0, 0.0
    p = j / total
    pe = p.sum(axis=1, keepdims=True); pt = p.sum(axis=0, keepdims=True)
    mask = p > 0
    mi = np.sum(p[mask] * np.log(p[mask] / (pe @ np.ones_like(pt))[mask] / (np.ones_like(pe) @ pt)[mask]))
    ptv = pt.ravel(); h = -np.sum(ptv[ptv > 0] * np.log(ptv[ptv > 0]))
    return float(mi), float(h)


def conditional_nmi_general(expert, target, cond, E, n_target, n_cond, min_count=50):
    """NMI(Expert; target | cond) = I(E;target|cond)/H(target|cond), and the
    unconditional NMI(E;target) for reference. If the conditional collapses toward
    0, `target`'s routing information was explained away by `cond` (the axes are
    entangled and `cond` is the primary one)."""
    expert = np.asarray(expert); target = np.asarray(target); cond = np.asarray(cond)
    total = len(expert)
    I, H = 0.0, 0.0
    for c in range(n_cond):
        m = cond == c
        if m.sum() < min_count:
            continue
        pc = m.sum() / total
        mi, h = _mi_h(_joint(expert[m], target[m], E, n_target))
        I += pc * mi; H += pc * h
    ju, hu = _mi_h(_joint(expert, target, E, n_target))
    return dict(cond=(I / H if H > 0 else 0.0), uncond=(ju / hu if hu > 0 else 0.0))


def soft_nmi(topk_idx, topk_w, domain, E, D):
    """NMI using the gate-weighted (soft, top-k) routing distribution instead of
    the top-1 argmax. More sensitive: a token contributes its gate weights to all
    its selected experts. topk_idx/topk_w: [N, K]; domain: [N]."""
    topk_idx = np.asarray(topk_idx); topk_w = np.asarray(topk_w); domain = np.asarray(domain)
    K = topk_idx.shape[1]
    j = np.zeros((E, D), dtype=np.float64)
    dom_rep = np.repeat(domain, K)
    np.add.at(j, (topk_idx.ravel(), dom_rep), topk_w.ravel())
    return _nmi_from_joint(j)


def conditional_nmi(expert, domain, token_ids, E, D, min_count=30):
    """I(Expert;Domain | Token) / H(Domain | Token): domain information in routing
    *after* removing anything explainable by token identity. Compared against the
    unconditional NMI on the same token subset. If conditional stays high, routing
    is genuinely context/domain-driven, not a relabelling of token frequencies."""
    expert = np.asarray(expert); domain = np.asarray(domain); token_ids = np.asarray(token_ids)
    uniq, counts = np.unique(token_ids, return_counts=True)
    keep = uniq[counts >= min_count]
    sub = np.isin(token_ids, keep)
    e_s, d_s, t_s = expert[sub], domain[sub], token_ids[sub]
    total = len(t_s)
    if total == 0:
        return dict(conditional=0.0, unconditional=0.0, n_tokens=0)
    I_cond, H_cond = 0.0, 0.0
    used = 0
    for t in keep:
        m = t_s == t
        dt = d_s[m]
        if len(np.unique(dt)) < 2:
            continue
        used += 1
        pt = m.sum() / total
        j = _joint(e_s[m], dt, E, D)
        # I(E;D|T=t) and H(D|T=t)
        p = j / j.sum()
        pe = p.sum(axis=1, keepdims=True); pd = p.sum(axis=0, keepdims=True)
        mask = p > 0
        mi = np.sum(p[mask] * np.log(p[mask] / (pe @ np.ones_like(pd))[mask] / (np.ones_like(pe) @ pd)[mask]))
        pdv = pd.ravel(); hd = -np.sum(pdv[pdv > 0] * np.log(pdv[pdv > 0]))
        I_cond += pt * mi; H_cond += pt * hd
    return dict(
        conditional=float(I_cond / H_cond) if H_cond > 0 else 0.0,
        unconditional=_nmi_from_joint(_joint(e_s, d_s, E, D)),
        n_tokens=used,
    )


def permutation_test(group_a, group_b):
    """Exact two-sided permutation test on the difference of means between two
    small groups (e.g. NMI across seeds for two conditions). Returns (diff, p)."""
    from itertools import combinations
    a = np.asarray(group_a, dtype=float); b = np.asarray(group_b, dtype=float)
    pooled = np.concatenate([a, b]); na = len(a)
    obs = a.mean() - b.mean()
    idx = list(range(len(pooled)))
    count, total = 0, 0
    for combo in combinations(idx, na):
        mask = np.zeros(len(pooled), dtype=bool); mask[list(combo)] = True
        diff = pooled[mask].mean() - pooled[~mask].mean()
        if abs(diff) >= abs(obs) - 1e-12:
            count += 1
        total += 1
    return float(obs), float(count / total)


def _jsd(dists):
    """Jensen-Shannon divergence among a list of prob vectors (uniform weights)."""
    dists = np.stack(dists)
    m = dists.mean(axis=0)
    def kl(p, q):
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None)))
    return float(np.mean([kl(d, m) for d in dists]))


def token_context_divergence(expert, domain, token_ids, E, D,
                             min_count=30, top_tokens=300, seed=0):
    """Q2 control: do identical tokens route differently across domains?

    For frequent tokens present in >=2 domains, JSD of their per-domain expert
    distribution. Compared to a control where domain labels are shuffled among
    each token's occurrences (destroying any real domain effect).

    observed >> control  =>  routing is context/domain-driven beyond token identity.
    """
    expert = np.asarray(expert); domain = np.asarray(domain); token_ids = np.asarray(token_ids)
    rng = np.random.default_rng(seed)

    uniq, counts = np.unique(token_ids, return_counts=True)
    cand = uniq[counts >= min_count]
    # rank by frequency, keep the most frequent shared tokens
    order = np.argsort(-counts[np.isin(uniq, cand)])
    cand = cand[order][:top_tokens]

    obs, ctl, used = [], [], 0
    for t in cand:
        sel = token_ids == t
        e_t, d_t = expert[sel], domain[sel]
        present = [d for d in range(D) if (d_t == d).sum() >= min_count // 3]
        if len(present) < 2:
            continue
        used += 1
        dists = []
        for d in present:
            ed = e_t[d_t == d]
            h = np.bincount(ed, minlength=E).astype(np.float64)
            dists.append(h / h.sum())
        obs.append(_jsd(dists))
        # control: shuffle domain labels among this token's occurrences
        d_sh = rng.permutation(d_t)
        dists_c = []
        for d in present:
            ed = e_t[d_sh == d]
            if len(ed) == 0:
                continue
            h = np.bincount(ed, minlength=E).astype(np.float64)
            dists_c.append(h / h.sum())
        if len(dists_c) >= 2:
            ctl.append(_jsd(dists_c))
    return dict(
        n_tokens=used,
        observed_jsd=float(np.mean(obs)) if obs else 0.0,
        control_jsd=float(np.mean(ctl)) if ctl else 0.0,
    )
