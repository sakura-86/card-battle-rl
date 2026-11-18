"""Search-budget sweep: how many simulations does it take to beat a rule?

The headline experiment.  It answers a question that unit tests cannot: at what
point does *searching* start paying for itself compared with simply encoding the
insight as a rule?

Run it with::

    python -m card_battle_rl.experiments.budget_scan --games 60

Output is a table plus a CSV under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..search.baselines import Agent, GreedyBot, RulePlanner
from ..search.puct import PUCTAgent
from .match import match

DEFAULT_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _build_opponent(name: str) -> Agent:
    if name == "rule_planner":
        return RulePlanner()
    if name == "greedy":
        return GreedyBot()
    raise SystemExit(f"unknown opponent {name!r}; choose rule_planner or greedy")


def sweep(
    budgets: tuple[int, ...],
    *,
    games: int,
    seed: int,
    opponent_name: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for simulations in budgets:
        opponent = _build_opponent(opponent_name)
        agent = PUCTAgent(simulations, select_by="q")
        result = match(agent, opponent, games=games, seed=seed)
        low, high = result.interval()
        row = {
            "simulations": simulations,
            "games": result.games,
            "wins": result.wins,
            "losses": result.losses,
            "draws": result.draws,
            "win_rate": result.win_rate,
            "ci_low": low,
            "ci_high": high,
            "opponent": opponent_name,
        }
        rows.append(row)
        print(
            f"{simulations:>4} sims  {result.wins:>3}W-{result.losses:>3}L  "
            f"{result.win_rate:6.1%}  95% CI [{low:5.1%}, {high:5.1%}]"
        )
    return rows


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=60, help="games per budget point")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--opponent", default="rule_planner", choices=["rule_planner", "greedy"]
    )
    parser.add_argument(
        "--budgets",
        default=",".join(str(value) for value in DEFAULT_BUDGETS),
        help="comma separated simulation counts",
    )
    parser.add_argument("--out", default=None, help="CSV path override")
    args = parser.parse_args(argv)

    budgets = tuple(int(part) for part in args.budgets.split(",") if part.strip())
    print(f"PUCT (uniform prior) vs {args.opponent}, {args.games} games per point\n")
    rows = sweep(
        budgets, games=args.games, seed=args.seed, opponent_name=args.opponent
    )

    out_path = (
        Path(args.out)
        if args.out
        else RESULTS_DIR / f"budget_scan_{args.opponent}.csv"
    )
    _write_csv(rows, out_path)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
