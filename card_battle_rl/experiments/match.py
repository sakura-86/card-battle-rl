"""Match harness: paired, seat-swapped games with confidence intervals.

Two design choices matter for trusting the numbers:

* **Seat swapping.** Half the games give agent A the first seat, half give it to
  agent B.  The opening hand differs by seat, so comparing only one direction
  would measure the deal rather than the policy.
* **Paired seeds.** Both games in a pair share a seed, so the same shuffles are
  replayed with seats reversed.  This removes most of the variance that a small
  sample would otherwise drown in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..env.toy import ToyCardBattle
from ..search.baselines import Agent

MAX_STEPS_PER_GAME = 400


@dataclass
class MatchResult:
    """Outcome of ``agent_a`` versus ``agent_b``."""

    agent_a: str
    agent_b: str
    games: int
    wins: int
    losses: int
    draws: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def score(self) -> float:
        """Win rate counting draws as half."""
        if not self.games:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def interval(self, z: float = 1.96) -> tuple[float, float]:
        return wilson_interval(self.wins, self.games, z)

    def summary(self) -> str:
        low, high = self.interval()
        return (
            f"{self.agent_a} vs {self.agent_b}: "
            f"{self.wins}W-{self.losses}L-{self.draws}D over {self.games} "
            f"({self.win_rate:.1%}, 95% CI [{low:.1%}, {high:.1%}])"
        )


def wilson_interval(wins: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- behaves sanely at 0 and 1, unlike Wald."""
    if trials <= 0:
        return (0.0, 1.0)
    p = wins / trials
    denominator = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    margin = (
        z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _play_one(env: ToyCardBattle, agent: Agent, seed: int) -> int:
    """Play a single game with ``agent`` in the first seat; return the winner id."""
    state = env.reset(seed)
    steps = 0
    while not state.done:
        action = agent.choose(env, state)
        state = env.step(state, action)
        steps += 1
        if steps > MAX_STEPS_PER_GAME:
            raise RuntimeError(
                f"game did not terminate within {MAX_STEPS_PER_GAME} steps (seed={seed})"
            )
    return state.winner


def match(
    agent_a: Agent,
    agent_b: Agent,
    *,
    games: int = 100,
    seed: int = 0,
) -> MatchResult:
    """Run ``games`` seat-swapped, seed-paired games."""
    if games % 2:
        games += 1
    wins = losses = draws = 0
    pairs = games // 2

    for pair in range(pairs):
        pair_seed = seed + pair

        # A in the first seat.
        env_a = ToyCardBattle(opponent=agent_b)
        winner = _play_one(env_a, agent_a, pair_seed)
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

        # B in the first seat, A responding.
        env_b = ToyCardBattle(opponent=agent_a)
        winner = _play_one(env_b, agent_b, pair_seed)
        if winner == 1:  # A was the responder
            wins += 1
        elif winner == 0:
            losses += 1
        else:
            draws += 1

    return MatchResult(
        agent_a=agent_a.name,
        agent_b=agent_b.name,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
    )
