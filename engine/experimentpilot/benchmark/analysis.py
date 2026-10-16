"""Benchmark analysis: summaries, confidence intervals, paired tests, calibration, plots, full pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..core.variants import VARIANTS
from ..likelihood.estimate import estimate_library, noisy_or_validation
from .runner import BENCH_HYPOTHESES, DEFAULT_CONFIG, build_specs, compute_records, run_planners

SUMMARY_COLS = ["top1", "top3", "precision", "recall", "f1", "exact", "brier", "cost", "n_experiments", "risk_total",
                "risk_violations"]


def bootstrap_ci(x: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarise(df: pd.DataFrame, n_boot: int = 1000) -> list[dict]:
    out = []
    for pid, g in df.groupby("planner", sort=False):
        row: dict = {"planner": pid, "label": VARIANTS[pid].label, "n_runs": int(len(g))}
        for c in SUMMARY_COLS:
            src = g[g["n_truth"] >= 1] if c in ("top1", "top3") else g
            m, lo, hi = bootstrap_ci(src[c].to_numpy(), n_boot, seed=1)
            row[c], row[c + "_lo"], row[c + "_hi"] = m, lo, hi
        row["stopped_confident"] = float((g["stop"] == "confidence_reached").mean())
        out.append(row)
    return out


def expected_calibration_error(marginals: np.ndarray, truth: np.ndarray, bins: int = 10):
    p, t = marginals.ravel(), truth.ravel()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, rel = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc = float(p[m].mean()), float(t[m].mean())
            ece += m.mean() * abs(conf - acc)
            rel.append({"bin": b, "confidence": conf, "accuracy": acc, "count": int(m.sum())})
    return float(ece), rel


def calibration(df: pd.DataFrame) -> dict:
    res = {}
    for pid, g in df.groupby("planner", sort=False):
        m = np.array(g["marginals"].tolist()); t = np.array(g["truth_vec"].tolist())
        ece, rel = expected_calibration_error(m, t)
        res[pid] = {"ece": ece, "brier": float(((m - t) ** 2).mean()), "reliability": rel}
    return res


def paired_tests(df: pd.DataFrame, reference: str = "adaptive") -> list[dict]:
    out = []
    piv = {c: df.pivot_table(index="scenario", columns="planner", values=c, aggfunc="first") for c in ("cost", "top1", "f1")}
    for other in [p for p in df["planner"].unique() if p != reference]:
        for metric in ("cost", "top1", "f1"):
            d = (piv[metric][reference] - piv[metric][other]).dropna()
            if len(d) < 5 or np.allclose(d, 0):
                p = 1.0
            else:
                try:
                    p = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
                except ValueError:
                    p = 1.0
            out.append({"reference": reference, "other": other, "metric": metric, "n": int(len(d)),
                        "mean_difference": float(d.mean()), "wilcoxon_p": p,
                        "note": "paired by scenario; lower cost is better, higher top1/f1 is better"})
    return out


def agreement_matrix(df: pd.DataFrame) -> dict:
    """Fraction of scenarios in which two planners executed the identical experiment sequence."""
    paths = df.assign(path=df["path"].map(tuple)).pivot_table(index="scenario", columns="planner", values="path", aggfunc="first")
    pl = list(paths.columns)
    return {"planners": pl, "matrix": [[float((paths[a] == paths[b]).mean()) for b in pl] for a in pl]}


def accuracy_vs_budget(records, library, cfg, planners) -> list[dict]:
    rows = []
    for B in cfg["budgets"]:
        df = run_planners(records, library, cfg, planners, budget=float(B), stop_on_confidence=False)
        for pid, g in df.groupby("planner", sort=False):
            single = g[g["n_truth"] >= 1]
            m, lo, hi = bootstrap_ci(single["top1"].to_numpy(), cfg["bootstrap"], seed=2)
            f, flo, fhi = bootstrap_ci(g["f1"].to_numpy(), cfg["bootstrap"], seed=3)
            rows.append({"planner": pid, "budget": B, "top1": m, "top1_lo": lo, "top1_hi": hi, "f1": f, "f1_lo": flo, "f1_hi": fhi,
                         "mean_cost": float(g["cost"].mean())})
    return rows


def robustness(records, library, cfg) -> list[dict]:
    rows = []
    for sg in cfg["robustness_sigmas"]:
        df = run_planners(records, library, cfg, ["adaptive", "checklist", "random"], sigma=float(sg))
        for pid, g in df.groupby("planner", sort=False):
            single = g[g["n_truth"] >= 1]
            rows.append({"sigma": sg, "planner": pid, "top1": float(single["top1"].mean()), "f1": float(g["f1"].mean()),
                         "cost": float(g["cost"].mean()), "brier": float(g["brier"].mean())})
    return rows


def breakdown(df: pd.DataFrame, by: str) -> list[dict]:
    out = []
    for (k, pid), g in df.groupby([by, "planner"], sort=False):
        out.append({by: k, "planner": pid, "n": int(len(g)), "top1": float(g["top1"].mean()) if g["top1"].notna().any() else None,
                    "f1": float(g["f1"].mean()), "cost": float(g["cost"].mean())})
    return out


# ------------------------------------------------------------------ plots
def make_plots(result: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    labels = {p: VARIANTS[p].label for p in VARIANTS}
    curve = pd.DataFrame(result["accuracy_vs_budget"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for pid, g in curve.groupby("planner", sort=False):
        ax.plot(g["budget"], g["top1"], marker="o", ms=3, label=labels[pid], lw=2.2 if pid == "adaptive" else 1.2)
        ax.fill_between(g["budget"], g["top1_lo"], g["top1_hi"], alpha=0.12)
    ax.set(xlabel="Diagnostic budget (cost units)", ylabel="Top-1 accuracy", title="Accuracy at fixed budget (simulated benchmark)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "accuracy_vs_budget.png", dpi=150); plt.close(fig); saved.append("accuracy_vs_budget.png")
    s = pd.DataFrame(result["summary"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(s["label"], s["cost"], yerr=[s["cost"] - s["cost_lo"], s["cost_hi"] - s["cost"]], capsize=3, color="#3b82f6")
    ax.set(ylabel="Mean diagnostic cost", title="Cost to diagnosis (95% bootstrap CI)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "cost_to_diagnosis.png", dpi=150); plt.close(fig); saved.append("cost_to_diagnosis.png")
    cal = (result.get("calibration") or {}).get("adaptive")
    if cal:
        r = pd.DataFrame(cal["reliability"])
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.plot(r["confidence"], r["accuracy"], marker="o")
        ax.set(xlabel="Predicted marginal probability", ylabel="Observed frequency", title=f"Reliability (ECE {cal['ece']:.3f})")
        ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out_dir / "reliability.png", dpi=150); plt.close(fig); saved.append("reliability.png")
    if result.get("robustness"):
        rb = pd.DataFrame(result["robustness"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for pid, g in rb.groupby("planner", sort=False):
            ax.plot(g["sigma"], g["f1"], marker="o", label=labels[pid])
        ax.set(xlabel="Likelihood perturbation sigma (logit)", ylabel="Multi-cause F1", title="Robustness to misspecified likelihoods")
        ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_dir / "robustness.png", dpi=150); plt.close(fig); saved.append("robustness.png")
    return saved


# ------------------------------------------------------------------ orchestration
def run_full_benchmark(cfg: dict | None = None, out_dir: str | Path = "research/results", workers: int = 4,
                       progress=None, write_library_to: Path | None = None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    est_specs = build_specs(cfg["estimation_datasets"], cfg["estimation_seeds"], cfg, pairs=True)
    eval_specs = build_specs(cfg["evaluation_datasets"], cfg["evaluation_seeds"], cfg, pairs=True)
    n_total = len(est_specs) + len(eval_specs)
    base = 0

    def prog(stage_off):
        return (lambda d, n: progress(stage_off + d, n_total, "computing experiment outcomes")) if progress else None

    est = compute_records(est_specs, workers, out / f"outcomes_{cfg['name']}_estimation.jsonl", prog(0))
    ev = compute_records(eval_specs, workers, out / f"outcomes_{cfg['name']}_evaluation.jsonl", prog(len(est_specs)))
    assert not ({r["id"] for r in est} & {r["id"] for r in ev}), "estimation/evaluation leakage"
    library = estimate_library(est, BENCH_HYPOTHESES, cfg["pseudo_count"],
                               provenance={"dataset_names": cfg["estimation_datasets"], "seeds": cfg["estimation_seeds"],
                                           "benchmark": cfg["name"]})
    if write_library_to:
        write_library_to.write_text(library.to_json(), encoding="utf8")
    (out / f"likelihood_{cfg['name']}.json").write_text(library.to_json(), encoding="utf8")
    if progress:
        progress(n_total, n_total, "evaluating planners")
    df = run_planners(ev, library, cfg)
    result = {
        "config": cfg, "simulated": True,
        "disclaimer": "All results come from simulated fault injection on built-in datasets; they do not demonstrate real-world gains.",
        "n_estimation_scenarios": len(est), "n_evaluation_scenarios": len(ev),
        "summary": summarise(df, cfg["bootstrap"]), "calibration": calibration(df),
        "paired_tests": paired_tests(df), "agreement": agreement_matrix(df),
        "by_fault": breakdown(df[df["n_truth"] <= 1], "truth"), "by_dataset": breakdown(df, "dataset"),
        "noisy_or_validation": noisy_or_validation(est, library, BENCH_HYPOTHESES),
        "accuracy_vs_budget": accuracy_vs_budget(ev, library, cfg, cfg["planners"]),
        "robustness": robustness(ev, library, cfg),
        "stop_reasons": df.groupby(["planner", "stop"]).size().reset_index(name="n").to_dict("records"),
    }
    # held-out-dataset generalisation (likelihoods estimated on synthetic data, evaluated on real breast_cancer data)
    bc = df[df["dataset"] == "breast_cancer"]
    if len(bc):
        result["unseen_dataset"] = {"dataset": "breast_cancer", "summary": summarise(bc, cfg["bootstrap"])}
    df.drop(columns=["marginals", "truth_vec"]).assign(path=df["path"].map(lambda p: ">".join(p))).to_csv(
        out / f"runs_{cfg['name']}.csv", index=False)
    result["plots"] = make_plots(result, out / "plots")
    result["elapsed_s"] = round(time.time() - t0, 1)
    (out / f"results_{cfg['name']}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf8")
    return result
