"""Built-in datasets (offline). All generators return the same schema:

record_id, event_time, <features...>, aux_signal, target
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATASETS = {
    "synthetic_credit": {"description": "Synthetic credit-default style table with numeric and categorical features.",
                         "learner": "logreg", "n_train": 2400, "n_ref": 1000, "n_recent": 1000, "synthetic": True},
    "synthetic_churn": {"description": "Synthetic customer-churn style table (non-linear target, gradient boosting model).",
                        "learner": "hgb", "n_train": 2400, "n_ref": 1000, "n_recent": 1000, "synthetic": True},
    "breast_cancer": {"description": "scikit-learn bundled Wisconsin diagnostic breast cancer data (real, small).",
                      "learner": "logreg", "n_train": 269, "n_ref": 150, "n_recent": 150, "synthetic": False},
}


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _credit(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = np.clip(rng.normal(41, 12, n), 18, 80)
    income = rng.lognormal(10.6, 0.5, n)
    debt = rng.beta(2, 5, n)
    util = rng.beta(2, 3, n)
    late = rng.poisson(1.0, n).astype(float)
    tenure = rng.exponential(48, n)
    score = rng.normal(650, 70, n)
    region = rng.choice(["north", "south", "east", "west"], n, p=[0.3, 0.25, 0.25, 0.2])
    emp = rng.choice(["salaried", "self_employed", "contract"], n, p=[0.6, 0.2, 0.2])
    z = (-1.8 + 4.2 * debt + 3.0 * util + 0.8 * late - 0.012 * (score - 650) - 0.8 * (np.log(income) - 10.6)
         - 0.016 * (age - 41) + 0.6 * (region == "south") + 0.8 * (emp == "contract") + rng.normal(0, 0.5, n))
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    return pd.DataFrame({"age": age, "income": income, "debt_ratio": debt, "utilization": util, "num_late": late,
                         "tenure_months": tenure, "credit_score": score, "region": region, "employment": emp,
                         "aux_signal": 0.3 * y + rng.normal(0, 1, n), "target": y})


def _churn(n: int, rng: np.random.Generator) -> pd.DataFrame:
    spend = rng.gamma(4, 18, n)
    tenure = rng.exponential(30, n)
    calls = rng.poisson(2.0, n).astype(float)
    usage = rng.gamma(3, 6, n)
    discount = rng.beta(1.5, 6, n)
    device = rng.uniform(0, 5, n)
    sat = np.clip(rng.normal(3.4, 1.0, n), 1, 5)
    contract = rng.choice(["monthly", "annual", "two_year"], n, p=[0.55, 0.3, 0.15])
    plan = rng.choice(["basic", "plus", "premium"], n, p=[0.4, 0.4, 0.2])
    z = (-1.0 + 0.55 * calls - 1.4 * (sat - 3.4) - 0.022 * tenure + 1.6 * (contract == "monthly") - 1.3 * (contract == "two_year")
         + 1.2 * np.sin(usage / 6.0) - 3.0 * discount + 0.6 * (spend > 100) * (plan == "basic") + rng.normal(0, 0.5, n))
    y = (rng.random(n) < _sigmoid(z)).astype(int)
    return pd.DataFrame({"monthly_spend": spend, "tenure_months": tenure, "support_calls": calls, "usage_hours": usage,
                         "discount_rate": discount, "device_age": device, "satisfaction": sat, "contract": contract,
                         "plan": plan, "aux_signal": 0.3 * y + rng.normal(0, 1, n), "target": y})


def _breast(n: int, rng: np.random.Generator) -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df = d.frame.rename(columns=lambda c: c.replace(" ", "_")).copy()
    df["target"] = df["target"].astype(int)
    df = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    df.insert(len(df.columns) - 1, "aux_signal", 0.3 * df["target"].to_numpy() + rng.normal(0, 1, len(df)))
    return df


def make_base(name: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset '{name}'")
    df = {"synthetic_credit": _credit, "synthetic_churn": _churn, "breast_cancer": _breast}[name](n, rng)
    df.insert(0, "record_id", np.arange(len(df)))
    df.insert(1, "event_time", pd.date_range("2025-01-01", periods=len(df), freq="h").strftime("%Y-%m-%d %H:%M"))
    return df
