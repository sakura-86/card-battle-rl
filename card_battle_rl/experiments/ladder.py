"""Strength ladder: every agent pair in one reproducible table.

Each row is a paired, seat-swapped match with a Wilson interval.  Writes
``results/ladder.csv`` so README numbers have a traceable source.

    python -m card_battle_rl.experiments.ladder --games 60
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

from ..search.baselines import Agent, GreedyBot, RandomBot, RulePlanner
from ..search.puct import PUCTAgent
from .match import MatchResult, match

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

AgentFactory = Callable[[], Agent]

# (left name, left factory, right name, default games)
DEFAULT_PAIRS: tuple[tuple[str, AgentFactory, str, int], ...] = (
    ("random", lambda: RandomBot(1), "greedy", 120),
    ("random", lambda: RandomBot(2), "rule_planner", 120),
    ("greedy", GreedyBot, "rule_planner", 200),
    ("puct_64", lambda: PUCTAgent(64, select_by="q"), "greedy", 60),
    ("puct_64", lambda: PUCTAgent(64, select_by="q"), "rule_planner", 120),
)


def run_pairs(
    games_override: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for left_name, left_factory, right_name, default_games in DEFAULT_PAIRS:
        games = games_override or default_games
        result: MatchResult = match(left_factory(), right_factory(right_name), games=games)
        low, high = result.interval()
        rows.append(
            {
                "agent_a": left_name,
                "agent_b": right_name,
                "games": result.games,
                "wins": result.wins,
                "losses": result.losses,
                "draws": result.draws,
                "win_rate": round(result.win_rate, 4),
                "ci_low": round(low, 4),
                "ci_high": round(high, 4),
            }
        )
        print(result.summary())
    return rows


def right_factory(name: str) -> Agent:
    if name == "greedy":
        return GreedyBot()
    if name == "rule_planner":
        return RulePlanner()
    raise SystemExit(f"unknown opponent {name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=0, help="override games per pair")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = run_pairs(args.games or None)
    out_path = Path(args.out) if args.out else RESULTS_DIR / "ladder.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
