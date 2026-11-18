"""Hand-written baselines: the opponents that search has to beat.

``RandomBot`` and ``GreedyBot`` are controls.  ``RulePlanner`` is the interesting
one: it encodes the ordering insight directly ("buff before you swing") in a few
dozen lines of rules, with no search at all.

The central question this repository asks is whether a PUCT agent, given a
comparable amount of compute, can outperform that rule.
"""

from __future__ import annotations

import random
from typing import Protocol

from ..env.core import END_TURN, Action, Card, CardKind, GameState
from ..env.toy import ToyCardBattle


class Agent(Protocol):
    """Anything that can pick a move."""

    name: str

    def choose(self, env: ToyCardBattle, state: GameState) -> Action: ...


class RandomBot:
    """Uniformly random legal move.  Establishes the floor."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def choose(self, env: ToyCardBattle, state: GameState) -> Action:
        return self._rng.choice(env.legal_actions(state))


class GreedyBot:
    """Always plays the card with the highest immediate damage.

    This is the natural "search one step and stop" agent, and it is exactly the
    behaviour that ``Focus`` is designed to punish: an un-buffed ``Strike`` looks
    better than ``Focus`` right up until you count the follow-ups.
    """

    name = "greedy"

    def choose(self, env: ToyCardBattle, state: GameState) -> Action:
        best_index = -1
        best_score = float("-inf")
        for index, card in enumerate(state.me.hand):
            if card.cost > state.me.mana:
                continue
            score = self._immediate_damage(state, card)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            return END_TURN
        return Action(kind="play", index=best_index)

    @staticmethod
    def _immediate_damage(state: GameState, card: Card) -> float:
        if card.kind is CardKind.ATTACK:
            return card.power + state.me.attack_bonus
        if card.kind is CardKind.MINION:
            return card.power
        return 0.0


class RulePlanner:
    """A rules-only turn planner.

    Priority order, evaluated against the *whole remaining turn* rather than a
    single step:

    1. If any attackers remain and a buff is affordable, buff first.
    2. Spend the rest on attacks, highest damage-per-mana first.
    3. Drop minions with leftover mana.
    4. Heal only when below the danger threshold.
    5. End the turn.

    There is no lookahead and no evaluation function -- just ordering rules.
    """

    name = "rule_planner"

    DANGER_HP = 12

    def choose(self, env: ToyCardBattle, state: GameState) -> Action:
        hand = state.me.hand
        affordable = [
            (index, card) for index, card in enumerate(hand) if card.cost <= state.me.mana
        ]
        if not affordable:
            return END_TURN

        attackers = [
            card for _, card in affordable if card.kind is CardKind.ATTACK
        ]
        buffs = [(index, card) for index, card in affordable if card.kind is CardKind.BUFF]

        # Rule 1: buff before attacking, but only if it pays for itself.
        if buffs and attackers:
            gained = sum(card.power for card in attackers)
            index, buff = buffs[0]
            if buff.power * len(attackers) >= buff.cost * 2 and gained > 0:
                return Action(kind="play", index=index)

        # Rule 2: best attack per mana.
        attacks = [(i, c) for i, c in affordable if c.kind is CardKind.ATTACK]
        if attacks:
            index, _ = max(
                attacks,
                key=lambda pair: (
                    (pair[1].power + state.me.attack_bonus) / max(pair[1].cost, 1),
                    -pair[0],
                ),
            )
            return Action(kind="play", index=index)

        # Rule 3: minions.
        minions = [(i, c) for i, c in affordable if c.kind is CardKind.MINION]
        if minions:
            index, _ = max(minions, key=lambda pair: (pair[1].power, -pair[0]))
            return Action(kind="play", index=index)

        # Rule 4: heal when in danger.
        if state.me.hero_hp <= self.DANGER_HP:
            heals = [(i, c) for i, c in affordable if c.kind is CardKind.HEAL]
            if heals:
                index, _ = max(heals, key=lambda pair: (pair[1].power, -pair[0]))
                return Action(kind="play", index=index)

        # Rule 5: cycle cheap draw spells with spare mana.
        draws = [(i, c) for i, c in affordable if c.kind is CardKind.DRAW]
        if draws:
            index, _ = min(draws, key=lambda pair: pair[1].cost)
            return Action(kind="play", index=index)

        return END_TURN
