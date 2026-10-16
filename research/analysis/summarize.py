"""Print a Markdown summary of a benchmark results file.

    python research/analysis/summarize.py research/results/results_default-v1.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    r = json.loads(Path(path).read_text(encoding="utf8"))
    print(f"# Benchmark summary ({'SIMULATED' if r.get('simulated') else 'unknown provenance'})\n")
    print(r.get("disclaimer", ""), "\n")
    print(f"Evaluation scenarios: {r.get('n_evaluation_scenarios', r.get('n_scenarios'))}; estimation scenarios: {r.get('n_estimation_scenarios', 'n/a')}\n")
    print("| Planner | Top-1 [95% CI] | Top-3 | F1 | Cost [95% CI] | # exp | Brier | Cap violations |")
    print("|---|---|---|---|---|---|---|---|")
    for s in r["summary"]:
        print(f"| {s['label']} | {s['top1']:.3f} [{s['top1_lo']:.3f}, {s['top1_hi']:.3f}] | {s['top3']:.3f} | {s['f1']:.3f} | "
              f"{s['cost']:.2f} [{s['cost_lo']:.2f}, {s['cost_hi']:.2f}] | {s['n_experiments']:.2f} | {s['brier']:.3f} | {s['risk_violations']:.2f} |")
    print("\n## Accuracy at fixed budget (top-1)\n")
    budgets = sorted({row["budget"] for row in r["accuracy_vs_budget"]})
    planners = list(dict.fromkeys(row["planner"] for row in r["accuracy_vs_budget"]))
    print("| Planner | " + " | ".join(f"B={b:g}" for b in budgets) + " |")
    print("|---|" + "---|" * len(budgets))
    for p in planners:
        vals = {row["budget"]: row["top1"] for row in r["accuracy_vs_budget"] if row["planner"] == p}
        print(f"| {p} | " + " | ".join(f"{vals[b]:.3f}" for b in budgets) + " |")
    print("\n## Paired tests (adaptive vs baseline)\n")
    for t in r.get("paired_tests", []):
        print(f"- {t['other']:13s} {t['metric']:5s} n={t['n']:4d} mean diff={t['mean_difference']:+.3f} Wilcoxon p={t['wilcoxon_p']:.3g}")
    cal = r.get("calibration", {}).get("adaptive")
    if cal:
        print(f"\nAdaptive planner calibration: ECE={cal['ece']:.3f}, Brier={cal['brier']:.3f}")
    if r.get("noisy_or_validation"):
        print("\nNoisy-OR validation on two-fault scenarios:", r["noisy_or_validation"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
