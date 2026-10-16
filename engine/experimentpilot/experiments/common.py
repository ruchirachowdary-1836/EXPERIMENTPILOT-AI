"""Shared statistics and data helpers for experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SMALL = 1e-4


def split_types(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    return num, cat


def psi_from_counts(ref_counts: np.ndarray, new_counts: np.ndarray) -> float:
    r = (ref_counts + 0.5) / (ref_counts.sum() + 0.5 * len(ref_counts))
    n = (new_counts + 0.5) / (new_counts.sum() + 0.5 * len(new_counts))
    return float(np.sum((n - r) * np.log(n / r)))


def numeric_psi(ref: np.ndarray, new: np.ndarray, bins: int = 10) -> float:
    ref, new = ref[np.isfinite(ref)], new[np.isfinite(new)]
    if len(ref) < 10 or len(new) < 10:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant reference
        return 0.0 if len(np.unique(new)) == 1 and np.unique(new)[0] == edges[0] else 5.0
    edges[0], edges[-1] = -np.inf, np.inf
    return psi_from_counts(np.histogram(ref, edges)[0].astype(float), np.histogram(new, edges)[0].astype(float))


def categorical_psi(ref: pd.Series, new: pd.Series) -> float:
    cats = sorted(set(ref.dropna().astype(str)) | set(new.dropna().astype(str)))
    rc = ref.dropna().astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(float)
    nc = new.dropna().astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(float)
    return psi_from_counts(rc, nc)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else 0.0


def cramers_v(ref: pd.Series, new: pd.Series) -> tuple[float, float]:
    cats = sorted(set(ref.dropna().astype(str)) | set(new.dropna().astype(str)))
    rc = ref.dropna().astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    nc = new.dropna().astype(str).value_counts().reindex(cats, fill_value=0).to_numpy()
    table = np.vstack([rc, nc])[:, table_keep(rc, nc)]
    if table.shape[1] < 2:
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))), float(p)


def table_keep(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) > 0


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    p = np.array([1.0 if (v is None or not np.isfinite(v)) else v for v in pvals], dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def design_matrix(fit_df: pd.DataFrame, cols: list[str], *dfs: pd.DataFrame, max_levels: int = 20):
    """Impute + standardise numerics and one-hot categoricals using ``fit_df`` statistics."""
    num, cat = split_types(fit_df, cols)
    outs = []
    med = {c: float(np.nanmedian(fit_df[c].to_numpy(float))) if fit_df[c].notna().any() else 0.0 for c in num}
    sd = {c: float(np.nanstd(fit_df[c].to_numpy(float))) or 1.0 for c in num}
    mu = {c: float(np.nanmean(fit_df[c].to_numpy(float))) if fit_df[c].notna().any() else 0.0 for c in num}
    levels = {c: list(fit_df[c].astype(str).value_counts().index[:max_levels]) for c in cat}
    for df in dfs:
        parts = []
        for c in num:
            v = df[c].to_numpy(float) if c in df else np.full(len(df), np.nan)
            v = np.where(np.isfinite(v), v, med[c])
            parts.append(((v - mu[c]) / sd[c])[:, None])
        for c in cat:
            s = df[c].astype(str) if c in df else pd.Series([""] * len(df))
            parts.append(np.column_stack([(s == lv).to_numpy(float) for lv in levels[c]]))
        outs.append(np.hstack(parts) if parts else np.zeros((len(df), 0)))
    return outs if len(outs) > 1 else outs[0]


def encode_labels(ref_y: pd.Series, *others: pd.Series):
    classes = sorted(pd.concat([ref_y] + list(others)).dropna().astype(str).unique())
    idx = {c: i for i, c in enumerate(classes)}
    return classes, [s.astype(str).map(idx) for s in (ref_y, *others)]


def accuracy(y: pd.Series, p: pd.Series) -> float:
    m = y.notna() & p.notna()
    return float((y[m].astype(str).to_numpy() == p[m].astype(str).to_numpy()).mean()) if m.any() else float("nan")


def macro_f1(y: pd.Series, p: pd.Series) -> float:
    m = y.notna() & p.notna()
    yy, pp = y[m].astype(str).to_numpy(), p[m].astype(str).to_numpy()
    f = []
    for c in np.unique(yy):
        tp = np.sum((yy == c) & (pp == c)); fp = np.sum((yy != c) & (pp == c)); fn = np.sum((yy == c) & (pp != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def auc_binary(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC; y in {0,1}."""
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    n1, n0 = int((yy == 1).sum()), int((yy == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(s)
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
