"""Run the ExperimentPilot fault-injection benchmark.

    python research/benchmark/run_benchmark.py --config research/configs/default.json --workers 6
    python research/benchmark/run_benchmark.py --quick          # tiny smoke run

Outputs (research/results/): outcomes_*.jsonl (cached measured outcomes), likelihood_*.json,
runs_*.csv, results_*.json and plots/*.png. With --install-library the estimated likelihood tables
become the default tables shipped with the engine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))


def main() -> None:
    from experimentpilot.benchmark import DEFAULT_CONFIG, run_full_benchmark
    from experimentpilot.likelihood import DEFAULT_TABLE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, help="JSON config overriding the defaults")
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="2 estimation + 2 evaluation seeds, 1 dataset")
    ap.add_argument("--install-library", action="store_true", help="write estimated tables as the engine default")
    a = ap.parse_args()
    cfg = dict(DEFAULT_CONFIG)
    if a.config:
        cfg.update(json.loads(a.config.read_text(encoding="utf8")))
    if a.quick:
        cfg.update(name="quick", estimation_datasets=["synthetic_credit"], estimation_seeds=[0, 1, 2],
                   evaluation_datasets=["synthetic_credit"], evaluation_seeds=[100, 101], budgets=[2, 4, 8, 12],
                   robustness_sigmas=[0.0, 1.0], bootstrap=200)
    def prog(d, n, msg):
        if d % 25 == 0 or d == n:
            print(f"[{d}/{n}] {msg}", flush=True)
    res = run_full_benchmark(cfg, a.out, a.workers, prog, DEFAULT_TABLE_PATH if a.install_library else None)
    print(f"\nDone in {res['elapsed_s']}s  (SIMULATED results)")
    print(f"{'planner':32s} top1   top3   F1     cost   n_exp  violations")
    for s in res["summary"]:
        print(f"{s['label']:32s} {s['top1']:.3f}  {s['top3']:.3f}  {s['f1']:.3f}  {s['cost']:.2f}  {s['n_experiments']:.2f}  {s['risk_violations']:.2f}")


if __name__ == "__main__":
    main()
