"""Train the PPO agent against the greedy baseline.

    python -m card_battle_rl.rl.train --iterations 40 --episodes 16

Training and evaluation use *disjoint seeds* on purpose: evaluating on the
training seeds would measure memorisation, not skill.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from ..env.toy import ToyCardBattle
from ..experiments.match import match
from ..search.baselines import GreedyBot, RulePlanner
from .agent import RLAgent
from .policy import PolicyValueNet
from .ppo import ppo_update
from .rollout import collect

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
TRAIN_SEED_STRIDE = 10_000
EVAL_SEED_STRIDE = 500_000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--episodes", type=int, default=16, help="episodes per iteration")
    parser.add_argument("--epochs", type=int, default=4, help="PPO passes per iteration")
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="ppo_vs_greedy")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    net = PolicyValueNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    env = ToyCardBattle(opponent=GreedyBot())

    rows: list[dict[str, object]] = []
    print(f"training {args.iterations} iterations x {args.episodes} episodes\n")

    for iteration in range(args.iterations):
        rollout = collect(
            env,
            net,
            episodes=args.episodes,
            seed_base=iteration * TRAIN_SEED_STRIDE,
        )
        stats = ppo_update(net, optimizer, rollout, epochs=args.epochs)

        if (iteration + 1) % args.eval_every == 0:
            agent = RLAgent(net, name=f"ppo@{iteration + 1}")
            vs_greedy = match(
                agent,
                GreedyBot(),
                games=args.eval_games,
                seed=EVAL_SEED_STRIDE + iteration * 1000,
            )
            vs_rule = match(
                agent,
                RulePlanner(),
                games=args.eval_games,
                seed=EVAL_SEED_STRIDE + iteration * 1000,
            )
            row = {
                "iteration": iteration + 1,
                "steps": len(rollout),
                "policy_loss": round(stats["policy_loss"], 4),
                "value_loss": round(stats["value_loss"], 4),
                "entropy": round(stats["entropy"], 4),
                "vs_greedy": round(vs_greedy.win_rate, 4),
                "vs_greedy_low": round(vs_greedy.interval()[0], 4),
                "vs_rule": round(vs_rule.win_rate, 4),
                "vs_rule_low": round(vs_rule.interval()[0], 4),
            }
            rows.append(row)
            print(
                f"iter {iteration + 1:>3}  steps {len(rollout):>5}  "
                f"pi_loss {stats['policy_loss']:+.3f}  v_loss {stats['value_loss']:.3f}  "
                f"H {stats['entropy']:.3f}  "
                f"vs_greedy {vs_greedy.win_rate:5.1%}  vs_rule {vs_rule.win_rate:5.1%}"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"{args.tag}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {csv_path}")

    checkpoint = RESULTS_DIR / f"{args.tag}.pt"
    torch.save(net.state_dict(), checkpoint)
    print(f"wrote {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
