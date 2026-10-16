"""Distribution-level experiments: drift, data-quality audit, class prior, pipeline replay."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .base import Experiment, ExperimentContext, ExperimentError, Finding, level_from
from .common import (accuracy, categorical_psi, cohens_d, cramers_v, holm, numeric_psi, split_types)


def _psi_label(v: float) -> str:
    return "major" if v >= 0.25 else "moderate" if v >= 0.10 else "stable"


class FeatureDrift(Experiment):
    id = "feature_drift"
    name = "Feature-distribution comparison"
    description = ("Per-feature PSI, two-sample Kolmogorov-Smirnov / chi-square tests (Holm-adjusted) and effect "
                   "sizes between reference and recent data, aggregated into a drift share.")
    cost, risk = 1.0, 0.0
    requires = ("features",)
    outcomes = ("no_shift", "moderate_shift", "severe_shift")
    limitations = ("Detects that distributions differ, not why: drift, preprocessing changes and upstream data "
                   "problems all shift feature distributions.",
                   "Univariate: multivariate shifts that keep marginals intact are not detected.")

    def _run(self, ctx: ExperimentContext):
        ref, rec = ctx.reference, ctx.recent
        feats = [f for f in ctx.feature_names() if f in rec.columns]
        if not feats:
            raise ExperimentError("no shared features to compare")
        num, cat = split_types(ref, feats)
        rows, pvals = [], []
        for f in num:
            a, b = ref[f].to_numpy(float), rec[f].to_numpy(float)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 10 or len(b) < 10:
                continue
            ks = stats.ks_2samp(a, b)
            rows.append({"feature": f, "type": "numeric", "psi": numeric_psi(a, b), "statistic": float(ks.statistic),
                         "test": "KS", "p_value": float(ks.pvalue), "effect_size": cohens_d(a, b),
                         "ref_mean": float(a.mean()), "recent_mean": float(b.mean())})
            pvals.append(float(ks.pvalue))
        for f in cat:
            v, p = cramers_v(ref[f], rec[f])
            rows.append({"feature": f, "type": "categorical", "psi": categorical_psi(ref[f], rec[f]), "statistic": v,
                         "test": "chi2", "p_value": p, "effect_size": v, "ref_mean": None, "recent_mean": None})
            pvals.append(p)
        if not rows:
            raise ExperimentError("not enough non-missing values to compare distributions")
        adj = holm(pvals)
        for r, pa in zip(rows, adj):
            r["p_adjusted"] = pa
            r["flagged"] = bool(r["psi"] >= 0.10 and pa < 0.01)
            r["severity"] = _psi_label(r["psi"])
        rows.sort(key=lambda r: -(r["psi"] if np.isfinite(r["psi"]) else 0))
        n_flag = sum(r["flagged"] for r in rows)
        share = n_flag / len(rows)
        max_psi = max(r["psi"] for r in rows if np.isfinite(r["psi"]))
        level = 2 if (share >= 0.4 or max_psi >= 1.0) else 1 if (share >= 0.1 or max_psi >= 0.25) else 0
        findings = [Finding("alert" if r["psi"] >= 0.25 else "warning",
                            f"{r['feature']}: PSI {r['psi']:.2f}, {r['test']} statistic {r['statistic']:.2f}")
                    for r in rows if r["flagged"]][:5]
        interp = (f"{n_flag} of {len(rows)} features ({share:.0%}) show a material distribution shift "
                  f"(PSI >= 0.10 and Holm-adjusted p < 0.01); the largest PSI is {max_psi:.2f}. "
                  + ("This is consistent with covariate drift, but also with a preprocessing change or upstream "
                     "data problem." if level else "No material feature shift was found."))
        return self.result(share, level, stats={"n_features": len(rows), "n_flagged": n_flag, "drift_share": share,
                                                "max_psi": max_psi, "mean_psi": float(np.mean([r["psi"] for r in rows])),
                                                "top_feature": rows[0]["feature"]},
                           tables={"features": rows}, findings=findings, interpretation=interp)


class DataQualityAudit(Experiment):
    id = "data_quality_audit"
    name = "Missingness, schema and range audit"
    description = ("Compares null rates, dtypes, value ranges, category sets, constant/stale columns and repeated "
                   "sentinel values between reference and recent data.")
    cost, risk = 1.0, 0.0
    requires = ("features",)
    outcomes = ("clean", "minor_issues", "serious_issues")
    limitations = ("Rule-based thresholds; semantic errors such as wrong-but-plausible values are not detected.",)

    def _run(self, ctx: ExperimentContext):
        ref, rec = ctx.reference, ctx.recent
        feats_ref = ctx.feature_names()
        findings, rows, schema = [], [], []
        missing_cols = [c for c in feats_ref if c not in rec.columns]
        new_cols = [c for c in rec.columns if c not in ref.columns and c not in
                    {ctx.mapping.target, ctx.mapping.prediction, ctx.mapping.timestamp, ctx.mapping.identifier}]
        for c in missing_cols:
            schema.append({"issue": "missing_column", "column": c})
        for c in new_cols:
            schema.append({"issue": "new_column", "column": c})
        critical = bool(schema)
        n_issue_features = 0
        for f in [c for c in feats_ref if c in rec.columns]:
            a, b = ref[f], rec[f]
            issues, crit = [], False
            n0, n1 = len(a), len(b)
            r0, r1 = float(a.isna().mean()), float(b.isna().mean())
            if n1 and r1 - r0 >= 0.02:
                pp = (r0 * n0 + r1 * n1) / (n0 + n1)
                z = (r1 - r0) / np.sqrt(max(pp * (1 - pp) * (1 / n0 + 1 / n1), 1e-12))
                if z > 3.5:
                    issues.append(f"null rate rose {r0:.1%} -> {r1:.1%}")
                    crit = crit or (r1 - r0 >= 0.10)
            a_num, b_num = pd.api.types.is_numeric_dtype(a), pd.api.types.is_numeric_dtype(b)
            if a_num != b_num:
                issues.append(f"dtype changed ({a.dtype} -> {b.dtype})"); crit = True
            oor = spike = unseen = float("nan")
            if a_num and b_num:
                av, bv = a.to_numpy(float), b.to_numpy(float)
                av, bv = av[np.isfinite(av)], bv[np.isfinite(bv)]
                if len(av) >= 20 and len(bv) >= 20:
                    lo, hi = np.quantile(av, [0.001, 0.999])
                    span = (hi - lo) or 1.0
                    oor = float(((bv < lo - 0.05 * span) | (bv > hi + 0.05 * span)).mean())
                    base = float(((av < lo - 0.05 * span) | (av > hi + 0.05 * span)).mean())
                    if oor - base >= 0.02:
                        issues.append(f"{oor:.1%} of values outside the reference range")
                        crit = crit or oor >= 0.20
                    vals, counts = np.unique(np.round(bv, 6), return_counts=True)
                    top = int(np.argmax(counts)); share = counts[top] / len(bv)
                    ref_share = float((np.round(av, 6) == vals[top]).mean())
                    spike = float(share)
                    if share >= 0.05 and ref_share <= 0.005:
                        issues.append(f"repeated value {vals[top]:g} in {share:.1%} of rows (possible sentinel/default)")
                        crit = crit or share >= 0.15
                    if np.unique(bv).size == 1 and np.unique(av).size > 2:
                        issues.append("column is constant in recent data (stale feature?)"); crit = True
            elif not a_num and not b_num:
                seen = set(a.dropna().astype(str))
                nb = b.dropna().astype(str)
                unseen = float((~nb.isin(seen)).mean()) if len(nb) else 0.0
                if unseen >= 0.01:
                    issues.append(f"{unseen:.1%} of values are categories unseen in reference"); crit = crit or unseen >= 0.1
            if issues:
                n_issue_features += 1
                findings.append(Finding("alert" if crit else "warning", f"{f}: " + "; ".join(issues)))
            critical = critical or crit
            rows.append({"feature": f, "null_ref": r0, "null_recent": r1, "dtype_ref": str(a.dtype),
                         "dtype_recent": str(b.dtype), "out_of_range_recent": oor, "top_value_share": spike,
                         "unseen_category_share": unseen, "issues": "; ".join(issues), "critical": crit})
        n_checked = max(len(rows), 1)
        share = n_issue_features / n_checked
        level = 2 if (critical or share >= 0.3) else 1 if (n_issue_features or schema) else 0
        for s in schema:
            findings.insert(0, Finding("alert", f"schema: {s['issue'].replace('_', ' ')} '{s['column']}'"))
        interp = (f"{n_issue_features} of {n_checked} features and {len(schema)} schema checks raised data-quality "
                  "flags." if level else "No missingness, schema, range or category anomalies were found.")
        return self.result(share, level, stats={"n_features": n_checked, "n_issue_features": n_issue_features,
                                                "n_schema_issues": len(schema), "issue_share": share,
                                                "max_null_increase": max([r["null_recent"] - r["null_ref"] for r in rows] or [0.0])},
                           tables={"features": rows, "schema": schema}, findings=findings[:8], interpretation=interp)


class ClassPriorComparison(Experiment):
    id = "class_prior"
    name = "Class-prior comparison"
    description = ("Compares label and prediction class frequencies (chi-square, total variation) and re-scores the "
                   "recent data under reference class priors.")
    cost, risk = 0.5, 0.0
    requires = ("labels_reference", "labels_recent")
    outcomes = ("stable", "moderate_shift", "severe_shift")
    limitations = ("Label-prior differences can also arise from label noise or sampling changes.",)

    def _run(self, ctx: ExperimentContext):
        t = ctx.mapping.target
        ref, rec = ctx.reference[t].dropna().astype(str), ctx.recent[t].dropna().astype(str)
        cls = sorted(set(ref) | set(rec))
        rc = ref.value_counts().reindex(cls, fill_value=0).to_numpy(float)
        nc = rec.value_counts().reindex(cls, fill_value=0).to_numpy(float)
        pr, pn = rc / rc.sum(), nc / nc.sum()
        tv = float(0.5 * np.abs(pr - pn).sum())
        keep = (rc + nc) > 0
        p = float(stats.chi2_contingency(np.vstack([rc, nc])[:, keep])[1]) if keep.sum() > 1 else 1.0
        level = 2 if (tv >= 0.12 and p < 1e-3) else 1 if (tv >= 0.05 and p < 1e-2) else 0
        st: dict = {"total_variation": tv, "chi2_p": p, "classes": cls, "ref_prior": pr.tolist(), "recent_prior": pn.tolist()}
        if "predictions_reference" in ctx.available_inputs() and "predictions_recent" in ctx.available_inputs():
            pp = ctx.mapping.prediction
            pcs = sorted(set(ctx.reference[pp].dropna().astype(str)) | set(ctx.recent[pp].dropna().astype(str)))
            a = ctx.reference[pp].astype(str).value_counts(normalize=True).reindex(pcs, fill_value=0).to_numpy()
            b = ctx.recent[pp].astype(str).value_counts(normalize=True).reindex(pcs, fill_value=0).to_numpy()
            st["prediction_total_variation"] = float(0.5 * np.abs(a - b).sum())
            # re-score the recent data under the reference class priors (per-class recall weighting)
            y, yhat = ctx.recent[t].astype(str), ctx.recent[pp].astype(str)
            m = ctx.recent[t].notna() & ctx.recent[pp].notna()
            recall = {c: float((yhat[m & (y == c)] == c).mean()) for c in cls if (m & (y == c)).any()}
            w = {c: pr[i] for i, c in enumerate(cls) if c in recall}
            if w:
                s = sum(w.values())
                st["recent_accuracy"] = accuracy(ctx.recent[t], ctx.recent[pp])
                st["recent_accuracy_under_reference_prior"] = float(sum(w[c] / s * recall[c] for c in w))
                st["reference_accuracy"] = accuracy(ctx.reference[t], ctx.reference[pp])
        rows = [{"class": c, "reference_share": float(a), "recent_share": float(b), "delta": float(b - a)}
                for c, a, b in zip(cls, pr, pn)]
        top = max(rows, key=lambda r: abs(r["delta"]))
        interp = (f"Label distributions differ by total variation {tv:.3f} (chi-square p = {p:.2g}); the largest change is "
                  f"class '{top['class']}' ({top['reference_share']:.1%} -> {top['recent_share']:.1%})."
                  if level else f"Class frequencies are stable (total variation {tv:.3f}, p = {p:.2g}).")
        return self.result(tv, level, stats=st, tables={"classes": rows},
                           findings=[Finding("alert" if level == 2 else "warning", interp)] if level else [],
                           interpretation=interp)


class PipelineReplay(Experiment):
    id = "pipeline_replay"
    name = "Replay through previous pipeline"
    description = ("Compares the features produced by the current preprocessing pipeline with the same raw records "
                   "processed by the previous pipeline version (supplied as a 'replay' dataset).")
    cost, risk = 2.0, 0.1
    requires = ("replay", "features")
    outcomes = ("identical", "minor_differences", "major_differences")
    limitations = ("Only as informative as the supplied replay dataset; both pipelines must receive identical raw records.",)

    def _run(self, ctx: ExperimentContext):
        rec, rep = ctx.recent, ctx.replay
        feats = [f for f in ctx.feature_names() if f in rec.columns and f in rep.columns]
        if not feats:
            raise ExperimentError("replay dataset shares no features with the recent dataset")
        idc = ctx.mapping.identifier
        aligned = len(rec) == len(rep)
        if aligned and idc and idc in rec.columns and idc in rep.columns:
            aligned = bool((rec[idc].astype(str).to_numpy() == rep[idc].astype(str).to_numpy()).all())
        rows = []
        num, cat = split_types(rec, feats)
        if aligned:
            for f in num:
                a, b = rec[f].to_numpy(float), rep[f].to_numpy(float)
                both_nan = np.isnan(a) & np.isnan(b)
                diff = np.abs(a - b)
                changed = (~both_nan) & ~(diff <= 1e-6 * (1 + np.abs(b)))
                sd = float(np.nanstd(b)) or 1.0
                rows.append({"feature": f, "type": "numeric", "changed_share": float(changed.mean()),
                             "mean_abs_diff_std": float(np.nanmean(np.where(changed, diff, 0.0)) / sd)})
            for f in cat:
                ch = rec[f].astype(str).to_numpy() != rep[f].astype(str).to_numpy()
                rows.append({"feature": f, "type": "categorical", "changed_share": float(ch.mean()), "mean_abs_diff_std": float(ch.mean())})
            for r in rows:
                r["flagged"] = bool(r["changed_share"] >= 0.05)
        else:
            for f in num:
                a, b = rec[f].to_numpy(float), rep[f].to_numpy(float)
                psi = numeric_psi(b, a)
                rows.append({"feature": f, "type": "numeric", "changed_share": float("nan"), "mean_abs_diff_std": abs(cohens_d(b, a)),
                             "psi": psi, "flagged": bool(psi >= 0.10)})
            for f in cat:
                psi = categorical_psi(rep[f], rec[f])
                rows.append({"feature": f, "type": "categorical", "changed_share": float("nan"), "mean_abs_diff_std": 0.0,
                             "psi": psi, "flagged": bool(psi >= 0.10)})
        n_flag = sum(r["flagged"] for r in rows)
        share = n_flag / len(rows)
        max_msd = max((r["mean_abs_diff_std"] for r in rows if np.isfinite(r["mean_abs_diff_std"])), default=0.0)
        level = 0 if n_flag == 0 else 2 if (share >= 0.25 or max_msd >= 0.5) else 1
        rows.sort(key=lambda r: -(r["mean_abs_diff_std"] if np.isfinite(r["mean_abs_diff_std"]) else 0))
        findings = [Finding("alert", f"{r['feature']}: outputs differ between pipeline versions"
                                     f" (mean |diff| = {r['mean_abs_diff_std']:.2f} sd)") for r in rows if r["flagged"]][:5]
        interp = (f"{n_flag} of {len(rows)} features are produced differently by the current and previous pipelines "
                  f"({'row-aligned' if aligned else 'distribution-level'} comparison). "
                  + ("A preprocessing or train-serving change is likely." if level else "The pipelines agree."))
        return self.result(share, level, stats={"aligned": aligned, "n_features": len(rows), "n_flagged": n_flag,
                                                "flagged_share": share, "max_mean_abs_diff_std": max_msd},
                           tables={"features": rows}, findings=findings, interpretation=interp)
