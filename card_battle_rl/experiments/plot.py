"""Render the headline figures from the CSV files in ``results/``.

    python -m card_battle_rl.experiments.plot

Produces ``results/budget_scan.png`` and ``results/ladder.png``.  Matplotlib is
an optional dependency: it is only needed for this module.
"""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_budget_scan(source: Path, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(source)
    simulations = [int(row["simulations"]) for row in rows]
    win_rates = [float(row["win_rate"]) * 100 for row in rows]
    lows = [float(row["ci_low"]) * 100 for row in rows]
    highs = [float(row["ci_high"]) * 100 for row in rows]

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.fill_between(simulations, lows, highs, alpha=0.18, color="#2b6cb0")
    axis.plot(simulations, win_rates, marker="o", color="#2b6cb0", linewidth=2)
    axis.axhline(50, linestyle="--", color="#718096", linewidth=1)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("PUCT simulations per decision (log scale)")
    axis.set_ylabel("Win rate vs rule planner (%)")
    axis.set_title("Extra search stops paying off after ~64 simulations")
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.25)
    for x, y in zip(simulations, win_rates):
        axis.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                      ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_ladder(source: Path, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(source)
    labels = [f"{row['agent_a']}\nvs {row['agent_b']}" for row in rows]
    win_rates = [float(row["win_rate"]) * 100 for row in rows]
    errors = [
        (
            (float(row["win_rate"]) - float(row["ci_low"])) * 100,
            (float(row["ci_high"]) - float(row["win_rate"])) * 100,
        )
        for row in rows
    ]
    lower = [error[0] for error in errors]
    upper = [error[1] for error in errors]

    figure, axis = plt.subplots(figsize=(7.6, 4.0))
    positions = range(len(rows))
    axis.bar(positions, win_rates, color="#4a5568", width=0.6)
    axis.errorbar(
        positions,
        win_rates,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#e53e3e",
        capsize=4,
    )
    axis.axhline(50, linestyle="--", color="#718096", linewidth=1)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylabel("Win rate (%)")
    axis.set_title("Agent strength ladder (95% Wilson intervals)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> int:
    scan = RESULTS_DIR / "budget_scan_rule_planner.csv"
    ladder = RESULTS_DIR / "ladder.csv"
    if scan.exists():
        plot_budget_scan(scan, RESULTS_DIR / "budget_scan.png")
        print(f"wrote {RESULTS_DIR / 'budget_scan.png'}")
    else:
        print(f"missing {scan}; run budget_scan first")
    if ladder.exists():
        plot_ladder(ladder, RESULTS_DIR / "ladder.png")
        print(f"wrote {RESULTS_DIR / 'ladder.png'}")
    else:
        print(f"missing {ladder}; run ladder first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
