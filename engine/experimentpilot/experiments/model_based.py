"""Experiments that fit small surrogate models or recompute metrics from supplied predictions.

No uploaded model artefact is ever deserialised: the model's behaviour enters only through
the prediction column supplied with the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from .base import Experiment, ExperimentContext, ExperimentError, Finding
from .common import (accuracy, auc_binary, design_matrix, encode_labels, holm, macro_f1, split_types)


def make_learner(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=60, max_depth=3, learning_rate=0.1, random_state=seed)
    return LogisticRegression(max_iter=300, C=1.0, random_state=seed)


def _clean_xy(df: pd.DataFrame, feats: list[str], target: str):
    d = df[df[target].notna()]
    return d, d[target].astype(str)


class LabelNoiseAudit(Experiment):
    id = "label_noise"
    name = "Label audit (confident learning)"
    description = ("Confident-learning-style estimate of the share of suspicious labels, computed with out-of-fold "
                   "probabilities on reference and on recent data separately; compares the two noise rates.")
    cost, risk = 3.0, 0.0
    requires = ("labels_reference", "labels_recent", "features")
    outcomes = ("no_excess_noise", "some_excess_noise", "high_excess_noise")
    limitations = ("A statistical indicator only: confident learning cannot tell label noise from genuinely hard or "
                   "overlapping examples, and cannot separate noise from a deliberate label-definition change.",
                   "Meaningful only if the surrogate model can fit the reference data reasonably well.",
                   "A leaky reference set makes reference labels look unusually clean, inflating the excess.")

    @staticmethod
    def _noise_rate(X: np.ndarray, y: np.ndarray, seed: int, learner: str) -> tuple[float, int]:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2 or counts.min() < 8:
            raise ExperimentError("each class needs at least 8 labelled rows for the label audit")
        folds = int(min(4, counts.min()))
        proba = cross_val_predict(make_learner(learner, seed), X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                                  method="predict_proba")
        t = np.array([proba[y == j, j].mean() for j in range(len(classes))])
        flagged = 0
        for i in range(len(y)):
            cand = [j for j in range(len(classes)) if proba[i, j] >= t[j]]
            if cand:
                j = max(cand, key=lambda k: proba[i, k])
                flagged += int(j != y[i])
        return flagged / len(y), len(y)

    def _run(self, ctx: ExperimentContext):
        tgt = ctx.mapping.target
        feats = ctx.feature_names()
        dr, yr = _clean_xy(ctx.reference, feats, tgt)
        dn, yn = _clean_xy(ctx.recent, feats, tgt)
        classes, (yr_i, yn_i) = encode_labels(yr, yn)
        Xr = design_matrix(dr, feats, dr)
        Xn = design_matrix(dr, feats, dn)  # same encoding; each set is modelled separately
        learner = ctx.config.get("learner", "logreg")
        rate_r, n_r = self._noise_rate(Xr, yr_i.to_numpy(int), ctx.seed, learner)
        rate_n, n_n = self._noise_rate(Xn, yn_i.to_numpy(int), ctx.seed, learner)
        excess = rate_n - rate_r
        se = np.sqrt(max(rate_r * (1 - rate_r) / n_r + rate_n * (1 - rate_n) / n_n, 1e-12))
        z = excess / se
        level = 2 if (excess >= 0.10 and z >= 3) else 1 if (excess >= 0.04 and z >= 2.5) else 0
        interp = (f"Estimated suspicious-label rate: {rate_r:.1%} in reference vs {rate_n:.1%} in recent data "
                  f"(excess {excess:+.1%}, z = {z:.1f}). "
                  + ("Recent labels look noisier than reference labels." if level else "No excess label noise detected."))
        return self.result(excess, level, stats={"noise_rate_reference": rate_r, "noise_rate_recent": rate_n,
                                                 "excess": excess, "z": float(z), "n_reference": n_r, "n_recent": n_n},
                           tables={"rates": [{"dataset": "reference", "suspicious_label_rate": rate_r, "n": n_r},
                                             {"dataset": "recent", "suspicious_label_rate": rate_n, "n": n_n}]},
                           findings=[Finding("alert" if level == 2 else "warning", interp)] if level else [],
                           interpretation=interp)


class LeakageScan(Experiment):
    id = "leakage_scan"
    name = "Leakage scan"
    description = ("Single-feature predictive power on reference vs recent data (AUC-based), retention of predictive "
                   "power relative to other features, and an optional temporal-split check.")
    cost, risk = 2.0, 0.0
    requires = ("labels_reference", "labels_recent", "features")
    outcomes = ("no_leak_signature", "possible_leak", "likely_leak")
    limitations = ("A feature that is very predictive on reference but not on recent data may be leaked, drifted, or "
                   "broken upstream; causal confirmation requires domain knowledge.",
                   "Needs at least one strongly predictive feature; weak leaks are not detected.")

    @staticmethod
    def _strength(df: pd.DataFrame, f: str, y01: np.ndarray, seed: int) -> float:
        if pd.api.types.is_numeric_dtype(df[f]):
            auc = auc_binary(df[f].to_numpy(float), y01)
        else:  # out-of-fold target encoding of a categorical feature
            s = df[f].astype(str).to_numpy()
            enc = np.zeros(len(s))
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(s, y01):
                means = pd.Series(y01[tr]).groupby(s[tr]).mean()
                enc[te] = pd.Series(s[te]).map(means).fillna(y01[tr].mean()).to_numpy()
            auc = auc_binary(enc, y01)
        return float(abs(auc - 0.5) * 2) if np.isfinite(auc) else 0.0

    def _run(self, ctx: ExperimentContext):
        tgt, feats = ctx.mapping.target, ctx.feature_names()
        dr, dn = ctx.reference[ctx.reference[tgt].notna()], ctx.recent[ctx.recent[tgt].notna()]
        classes = sorted(set(dr[tgt].astype(str)) | set(dn[tgt].astype(str)))
        pos = classes[-1]  # one-vs-rest on the last class
        yr, yn = (dr[tgt].astype(str) == pos).to_numpy(int), (dn[tgt].astype(str) == pos).to_numpy(int)
        rows = []
        for f in [f for f in feats if f in dn.columns]:
            sr, sn = self._strength(dr, f, yr, ctx.seed), self._strength(dn, f, yn, ctx.seed)
            rows.append({"feature": f, "strength_reference": sr, "strength_recent": sn,
                         "retention": float(np.clip(sn / sr, 0, 1.5)) if sr > 0.05 else None})
        inform = [r for r in rows if r["strength_reference"] >= 0.2]
        score, culprit = 0.0, None
        for r in inform:
            if r["strength_reference"] < 0.4:
                continue
            others = [o["retention"] for o in inform if o is not r and o["retention"] is not None]
            med = float(np.median(others)) if others else 1.0
            gap = med - r["retention"]
            r["retention_gap"] = gap
            if gap > score:
                score, culprit = gap, r["feature"]
        level = 2 if score >= 0.55 else 1 if score >= 0.25 else 0
        very = [r["feature"] for r in rows if r["strength_reference"] >= 0.96]
        st: dict = {"leak_score": score, "suspect_feature": culprit, "n_informative_features": len(inform),
                    "near_perfect_features": very}
        ts = ctx.mapping.timestamp
        if ts and ts in ctx.reference.columns and len(dr) >= 200:
            try:
                order = np.argsort(pd.to_datetime(dr[ts], errors="coerce").to_numpy().astype("datetime64[ns]"))
                d = dr.iloc[order]
                cut = int(len(d) * 0.6)
                X = design_matrix(d, feats, d)
                yy = (d[tgt].astype(str) == pos).to_numpy(int)
                if len(np.unique(yy[:cut])) > 1 and len(np.unique(yy[cut:])) > 1:
                    m = make_learner("logreg", ctx.seed).fit(X[:cut], yy[:cut])
                    temporal = float(m.score(X[cut:], yy[cut:]))
                    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.4, random_state=ctx.seed, stratify=yy)
                    rand = float(make_learner("logreg", ctx.seed).fit(Xtr, ytr).score(Xte, yte))
                    st.update({"temporal_split_accuracy": temporal, "random_split_accuracy": rand, "temporal_gap": rand - temporal})
            except Exception:
                st["temporal_split_accuracy"] = None
        rows.sort(key=lambda r: -r["strength_reference"])
        findings = []
        if culprit:
            cr = next(r for r in rows if r["feature"] == culprit)
            findings.append(Finding("alert" if level == 2 else "warning",
                                    f"{culprit}: single-feature strength {cr['strength_reference']:.2f} on reference but "
                                    f"{cr['strength_recent']:.2f} on recent data"))
        for f in very:
            findings.append(Finding("alert", f"{f}: near-perfect single-feature predictive power on reference data"))
        interp = ((f"'{culprit}' is strongly predictive on reference data but loses most of that power on recent data "
                   f"(retention gap {score:.2f}), the signature of a feature that leaked the label during training."
                   ) if level else "No feature shows a leak-like loss of predictive power.")
        return self.result(score, level, stats=st, tables={"features": rows}, findings=findings, interpretation=interp)


class ImportanceWeightedEval(Experiment):
    id = "importance_weighted_eval"
    name = "Importance-weighted re-evaluation"
    description = ("Reweights reference examples (domain-classifier density ratio) to resemble recent inputs and checks "
                   "whether the performance gap closes.")
    cost, risk = 2.0, 0.0
    requires = ("labels_reference", "labels_recent", "predictions_reference", "predictions_recent", "features")
    outcomes = ("gap_not_explained", "gap_partly_explained", "gap_mostly_explained")
    limitations = ("Assumes covariate shift only (P(y|x) unchanged) and sufficient overlap between the two inputs.",
                   "Also 'explains' gaps caused by pipeline changes or data-quality problems, because those shift inputs too.")

    def _run(self, ctx: ExperimentContext):
        t, p, feats = ctx.mapping.target, ctx.mapping.prediction, ctx.feature_names()
        ref = ctx.reference[ctx.reference[t].notna() & ctx.reference[p].notna()]
        rec = ctx.recent[ctx.recent[t].notna() & ctx.recent[p].notna()]
        Xr, Xn = design_matrix(ref, feats, ref, rec)
        X = np.vstack([Xr, Xn]); d = np.r_[np.zeros(len(Xr)), np.ones(len(Xn))]
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=ctx.seed)
        proba = cross_val_predict(clf, X, d, cv=StratifiedKFold(3, shuffle=True, random_state=ctx.seed), method="predict_proba")[:, 1]
        dom_auc = auc_binary(proba, d.astype(int))
        pr = np.clip(proba[: len(Xr)], 1e-3, 1 - 1e-3)
        w = np.clip(pr / (1 - pr) * (len(Xr) / len(Xn)), 0, 20.0)
        correct = (ref[t].astype(str).to_numpy() == ref[p].astype(str).to_numpy()).astype(float)
        acc_ref, acc_w = float(correct.mean()), float((w * correct).sum() / max(w.sum(), 1e-12))
        acc_rec = accuracy(rec[t], rec[p])
        gap = acc_ref - acc_rec
        ess = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
        explained = (acc_ref - acc_w) / gap if gap >= 0.02 else 0.0
        level = 0 if gap < 0.02 else 2 if explained >= 0.6 else 1 if explained >= 0.25 else 0
        interp = ((f"Performance fell {gap:.1%} (accuracy {acc_ref:.1%} -> {acc_rec:.1%}). Reweighting the reference data to match "
                   f"recent inputs gives {acc_w:.1%}, explaining {explained:.0%} of the gap.") if gap >= 0.02 else
                  f"There is no meaningful accuracy gap to explain ({acc_ref:.1%} vs {acc_rec:.1%}).")
        return self.result(explained, level, stats={"accuracy_reference": acc_ref, "accuracy_recent": acc_rec,
                                                    "accuracy_reference_reweighted": acc_w, "gap": gap,
                                                    "gap_explained_share": explained, "domain_auc": dom_auc,
                                                    "effective_sample_size": ess},
                           tables={"summary": [{"metric": "reference accuracy", "value": acc_ref},
                                               {"metric": "reference accuracy (reweighted to recent inputs)", "value": acc_w},
                                               {"metric": "recent accuracy", "value": acc_rec}]},
                           findings=[], interpretation=interp)


class SliceAnalysis(Experiment):
    id = "slice_analysis"
    name = "Slice analysis"
    description = ("Finds data slices (categorical levels, numeric quantile bins, missing-value slices) whose accuracy "
                   "dropped most relative to the overall drop.")
    cost, risk = 2.0, 0.0
    requires = ("labels_reference", "labels_recent", "predictions_reference", "predictions_recent", "features")
    outcomes = ("uniform_drop", "localized_drop", "severely_localized_drop")
    limitations = ("Many slices are tested; Holm adjustment controls false positives but small slices have little power.",
                   "Localisation points to where the problem is, not what caused it.")

    def _run(self, ctx: ExperimentContext):
        t, p, feats = ctx.mapping.target, ctx.mapping.prediction, ctx.feature_names()
        ref = ctx.reference[ctx.reference[t].notna() & ctx.reference[p].notna()].reset_index(drop=True)
        rec = ctx.recent[ctx.recent[t].notna() & ctx.recent[p].notna()].reset_index(drop=True)
        cr = (ref[t].astype(str) == ref[p].astype(str)).to_numpy()
        cn = (rec[t].astype(str) == rec[p].astype(str)).to_numpy()
        overall = float(cr.mean() - cn.mean())
        slices = []
        for f in [f for f in feats if f in rec.columns]:
            masks = []
            if pd.api.types.is_numeric_dtype(ref[f]):
                v = ref[f].to_numpy(float)
                edges = np.unique(np.nanquantile(v, [0.25, 0.5, 0.75]))
                lab = lambda arr: np.where(np.isnan(arr), -1, np.searchsorted(edges, arr, side="right"))
                lr_, ln_ = lab(v), lab(rec[f].to_numpy(float))
                for k in sorted(set(np.unique(lr_)) | set(np.unique(ln_))):
                    name = "missing" if k == -1 else f"bin {int(k) + 1}/{len(edges) + 1}"
                    masks.append((f, name, lr_ == k, ln_ == k))
            else:
                levels = ref[f].astype(str).value_counts().index[:12]
                for lv in levels:
                    masks.append((f, str(lv), (ref[f].astype(str) == lv).to_numpy(), (rec[f].astype(str) == lv).to_numpy()))
            for f_, name, mr, mn in masks:
                nr, nn = int(mr.sum()), int(mn.sum())
                if nr < 30 or nn < 30:
                    continue
                ar, an = float(cr[mr].mean()), float(cn[mn].mean())
                pp = (ar * nr + an * nn) / (nr + nn)
                z = (ar - an) / np.sqrt(max(pp * (1 - pp) * (1 / nr + 1 / nn), 1e-12))
                slices.append({"feature": f_, "slice": name, "n_reference": nr, "n_recent": nn, "accuracy_reference": ar,
                               "accuracy_recent": an, "drop": ar - an, "excess_drop": ar - an - overall,
                               "p_value": float(stats.norm.sf(z))})
        if not slices:
            raise ExperimentError("no slice had enough rows (>= 30) in both datasets")
        adj = holm([s["p_value"] for s in slices])
        for s, a in zip(slices, adj):
            s["p_adjusted"] = a
            s["significant"] = bool(a < 0.01 and s["excess_drop"] > 0)
        slices.sort(key=lambda s: -s["excess_drop"])
        sig = [s for s in slices if s["significant"]]
        score = max((s["excess_drop"] for s in sig), default=0.0)
        level = 2 if score >= 0.20 else 1 if score >= 0.08 else 0
        worst = sig[0] if sig else slices[0]
        interp = ((f"The worst slice is {worst['feature']} = {worst['slice']}: accuracy fell {worst['drop']:.1%} versus "
                   f"{overall:.1%} overall, so the degradation is concentrated rather than uniform.") if level else
                  f"Accuracy dropped {overall:.1%} overall with no slice standing out: the drop looks uniform.")
        return self.result(score, level, stats={"overall_drop": overall, "n_slices": len(slices), "n_significant": len(sig),
                                                "max_excess_drop": score},
                           tables={"slices": slices[:25]},
                           findings=[Finding("alert" if level == 2 else "warning",
                                             f"{s['feature']} = {s['slice']}: drop {s['drop']:.1%} (n={s['n_recent']})")
                                     for s in sig[:4]], interpretation=interp)


class RetrainOnRecent(Experiment):
    id = "retrain_recent"
    name = "Retrain on recent labelled sample"
    description = ("Trains a small surrogate model on a recent labelled sample (copy of the data) and checks whether "
                   "accuracy on held-out recent rows recovers compared with a surrogate trained on reference data.")
    cost, risk = 5.0, 0.2
    requires = ("labels_reference", "labels_recent", "features")
    outcomes = ("no_recovery", "partial_recovery", "strong_recovery")
    requires_approval = True
    limitations = ("Needs a few hundred labelled recent rows; the surrogate is a small model, not your production model.",
                   "Recovery shows the data supports a better model, which fits drift or leakage but not label noise.")

    def _run(self, ctx: ExperimentContext):
        t, feats = ctx.mapping.target, ctx.feature_names()
        dr = ctx.reference[ctx.reference[t].notna()]
        dn = ctx.recent[ctx.recent[t].notna()].reset_index(drop=True)
        if len(dn) < 200:
            raise ExperimentError("retraining needs at least 200 labelled recent rows")
        yn = dn[t].astype(str)
        classes, (yr_i, yn_i) = encode_labels(dr[t].astype(str), yn)
        yn_i = yn_i.to_numpy(int)
        if np.bincount(yn_i).min() < 5:
            raise ExperimentError("each class needs at least 5 recent labelled rows")
        learner = ctx.config.get("learner", "logreg")
        idx = np.arange(len(dn))
        ts = ctx.mapping.timestamp
        if ts and ts in dn.columns and pd.to_datetime(dn[ts], errors="coerce").notna().all():
            order = np.argsort(pd.to_datetime(dn[ts]).to_numpy().astype("datetime64[ns]"))
            cut = int(len(dn) * 0.7); tr, te = order[:cut], order[cut:]
            split = "temporal"
        else:
            tr, te = train_test_split(idx, test_size=0.3, random_state=ctx.seed, stratify=yn_i)
            split = "random"
        Xr, Xn = design_matrix(dr, feats, dr, dn)
        m_ref = make_learner(learner, ctx.seed).fit(Xr, yr_i.to_numpy(int))
        m_new = make_learner(learner, ctx.seed).fit(Xn[tr], yn_i[tr])
        acc_ref_model = float(m_ref.score(Xn[te], yn_i[te]))
        acc_new_model = float(m_new.score(Xn[te], yn_i[te]))
        rec = acc_new_model - acc_ref_model
        st = {"holdout_rows": int(len(te)), "train_rows": int(len(tr)), "split": split,
              "accuracy_reference_trained": acc_ref_model, "accuracy_recent_trained": acc_new_model, "recovery": rec}
        p = ctx.mapping.prediction
        if p and p in dn.columns and dn[p].notna().all():
            st["accuracy_production_model"] = accuracy(dn[t].iloc[te], dn[p].iloc[te])
        level = 2 if rec >= 0.10 else 1 if rec >= 0.03 else 0
        interp = (f"A surrogate trained on recent data scores {acc_new_model:.1%} on held-out recent rows vs {acc_ref_model:.1%} for "
                  f"one trained on reference data (recovery {rec:+.1%}). "
                  + ("The data supports a better model, which suggests drift or a train-serving mismatch rather than label noise."
                     if level else "Retraining on recent data does not help, which points away from drift and toward noisy labels or a metric problem."))
        return self.result(rec, level, stats=st,
                           tables={"summary": [{"model": "trained on reference", "holdout_accuracy": acc_ref_model},
                                               {"model": "trained on recent", "holdout_accuracy": acc_new_model}]},
                           interpretation=interp)


class MetricRecompute(Experiment):
    id = "metric_recompute"
    name = "Independent metric recomputation"
    description = "Recomputes the reported metric from raw predictions and labels and compares it with the reported value."
    cost, risk = 0.25, 0.0
    requires = ("labels_reference", "labels_recent", "predictions_reference", "predictions_recent", "reported_metrics")
    outcomes = ("matches_report", "minor_mismatch", "major_mismatch")
    limitations = ("Only validates the arithmetic of the reported metric given the supplied files, not their provenance.",)

    def _run(self, ctx: ExperimentContext):
        rm = ctx.reported_metrics or {}
        name = str(rm.get("metric", "accuracy"))
        fn = {"accuracy": accuracy, "macro_f1": macro_f1}.get(name)
        if fn is None:
            raise ExperimentError("reported metric must be 'accuracy' or 'macro_f1'")
        t, p = ctx.mapping.target, ctx.mapping.prediction
        rr, rn = fn(ctx.reference[t], ctx.reference[p]), fn(ctx.recent[t], ctx.recent[p])
        dr, dn = abs(rr - float(rm["reference"])), abs(rn - float(rm["recent"]))
        score = max(dr, dn)
        level = 2 if score >= 0.03 else 1 if score >= 0.01 else 0
        rows = [{"dataset": "reference", "reported": float(rm["reference"]), "recomputed": rr, "difference": dr},
                {"dataset": "recent", "reported": float(rm["recent"]), "recomputed": rn, "difference": dn}]
        interp = (f"Recomputed {name}: reference {rr:.3f} (reported {float(rm['reference']):.3f}), recent {rn:.3f} "
                  f"(reported {float(rm['recent']):.3f}); max discrepancy {score:.3f}. "
                  + ("The reported numbers cannot be reproduced from the supplied predictions and labels." if level
                     else "The reported metrics are reproducible."))
        return self.result(score, level, stats={"metric": name, "max_discrepancy": score, "recomputed_reference": rr,
                                                "recomputed_recent": rn},
                           tables={"comparison": rows},
                           findings=[Finding("alert" if level == 2 else "warning", interp)] if level else [],
                           interpretation=interp)
