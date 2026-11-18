"""Tests for evaluation, baselines and PUCT.

The evaluator tests are regression tests for two real bugs found during
development; both produced an agent that lost every game, and neither raised an
exception.
"""

from __future__ import annotations

import unittest

from card_battle_rl.env import END_TURN, Action, GameState, ToyCardBattle
from card_battle_rl.env.toy import BASE_HP, CARD_LIBRARY
from card_battle_rl.search.baselines import GreedyBot, RandomBot, RulePlanner
from card_battle_rl.search.evaluation import heuristic_value
from card_battle_rl.search.puct import PUCTAgent, policy_target, search


def build_state(hand: tuple[int, ...], mana: int) -> GameState:
    """Craft a state with a specific hand and no deck."""
    env = ToyCardBattle()
    base = env.reset(seed=0)
    cards = tuple(CARD_LIBRARY[card_id] for card_id in hand)
    me = base.me.__class__(hero_hp=BASE_HP, mana=mana, hand=cards)
    opponent = base.opponent.__class__(hero_hp=BASE_HP, mana=0)
    return GameState(me=me, opponent=opponent, turn=1)


class EvaluationSignTests(unittest.TestCase):
    """Regression: HP advantage used to be computed with the wrong sign.

    With ``opponent_hp - me_hp``, dealing damage *lowered* the score, so the
    search learned to end the turn immediately and lost 40 games in a row.
    """

    def test_dealing_damage_does_not_reduce_value(self) -> None:
        """Spending damage now must be scored at least as well as hoarding it.

        HR and potential are two ledgers for the same damage; the evaluator keeps
        them conserved, so this is neutral rather than positive.
        """
        env = ToyCardBattle()
        state = build_state((0, 0, 0), mana=3)
        before = heuristic_value(state)
        after = env.step(state, Action(kind="play", index=0))
        self.assertGreaterEqual(heuristic_value(after) + 1e-9, before)

    def test_buffing_with_followups_improves_value(self) -> None:
        """Focus before attackers must score above the same hand without Focus."""
        env = ToyCardBattle()
        with_focus = build_state((3, 0, 0, 0), mana=4)
        after_focus = env.step(with_focus, Action(kind="play", index=0))
        self.assertGreater(heuristic_value(after_focus), heuristic_value(with_focus))

    def test_taking_damage_worsens_value(self) -> None:
        env = ToyCardBattle()
        state = build_state((), mana=0)
        hurt = state.me.__class__(
            hero_hp=state.me.hero_hp - 5,
            mana=state.me.mana,
            hand=state.me.hand,
        )
        wounded = GameState(me=hurt, opponent=state.opponent, turn=1)
        self.assertLess(heuristic_value(wounded), heuristic_value(state))

    def test_ending_turn_without_playing_is_not_best(self) -> None:
        """Ending the turn must not beat actually playing a damage card."""
        env = ToyCardBattle()
        state = build_state((0, 0, 0), mana=3)
        played = env.step(state, Action(kind="play", index=0))
        closed = env.resolve_turn(state)
        self.assertGreater(heuristic_value(played), heuristic_value(closed))

    def test_closed_turn_has_no_remaining_potential(self) -> None:
        """Regression: leftover mana made a closed turn look like it still had plays."""
        env = ToyCardBattle()
        state = build_state((0, 0), mana=2)
        open_value = heuristic_value(state)
        closed_value = heuristic_value(env.resolve_turn(state))
        self.assertGreater(open_value, closed_value)


class BaselineLegalityTests(unittest.TestCase):
    def test_agents_only_return_legal_actions(self) -> None:
        env = ToyCardBattle()
        agents = [RandomBot(seed=1), GreedyBot(), RulePlanner(), PUCTAgent(8)]
        for seed in range(5):
            state = env.reset(seed=seed)
            legal = set(env.legal_actions(state))
            for agent in agents:
                action = agent.choose(env, state)
                self.assertIn(
                    action, legal, f"{agent.name} returned illegal {action} at seed {seed}"
                )


class PUCTTests(unittest.TestCase):
    def test_search_populates_root(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=0)
        root = search(env, state, simulations=16)
        self.assertTrue(root.children)
        self.assertEqual(root.visits, 16)

    def test_single_simulation_does_not_crash(self) -> None:
        env = ToyCardBattle()
        root = search(env, env.reset(seed=3), simulations=1)
        self.assertGreaterEqual(root.visits, 1)

    def test_puct_plays_a_card_when_damage_is_available(self) -> None:
        """Regression: the sign bug made PUCT always end the turn."""
        env = ToyCardBattle()
        agent = PUCTAgent(32, select_by="q")
        state = build_state((0, 0, 0), mana=3)  # three affordable Strikes
        action = agent.choose(env, state)
        self.assertEqual(action.kind, "play")

    def test_policy_target_sums_to_one(self) -> None:
        env = ToyCardBattle()
        root = search(env, env.reset(seed=0), simulations=16)
        target = policy_target(root)
        self.assertTrue(target)
        self.assertAlmostEqual(sum(target.values()), 1.0, places=6)

    def test_puct_does_not_recurse_when_opponent_is_puct(self) -> None:
        """A PUCT opponent must not recursively search inside its own search."""
        env = ToyCardBattle(opponent=PUCTAgent(4))
        agent = PUCTAgent(4)
        state = env.reset(seed=1)
        action = agent.choose(env, state)
        self.assertIn(action, set(env.legal_actions(state)))


if __name__ == "__main__":
    unittest.main()
