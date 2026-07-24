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
