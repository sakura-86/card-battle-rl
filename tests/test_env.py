"""Smoke tests for the toy environment."""

from __future__ import annotations

import unittest

from card_battle_rl.env import END_TURN, Action, CardKind, GameState, ToyCardBattle
from card_battle_rl.env.toy import BASE_HP, MAX_HAND, MAX_TURNS, CARD_LIBRARY


class ResetTests(unittest.TestCase):
    def test_reset_is_deterministic(self) -> None:
        env = ToyCardBattle()
        first = env.reset(seed=7)
        second = env.reset(seed=7)
        self.assertEqual(first, second)

    def test_different_seeds_differ(self) -> None:
        env = ToyCardBattle()
        self.assertNotEqual(env.reset(seed=1), env.reset(seed=2))

    def test_initial_shape(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=3)
        self.assertEqual(state.me.hero_hp, BASE_HP)
        self.assertEqual(state.opponent.hero_hp, BASE_HP)
        self.assertEqual(len(state.me.hand), 4)
        self.assertEqual(len(state.opponent.hand), 4)
        self.assertEqual(state.me.mana, 1)
        self.assertFalse(state.done)


class LegalActionTests(unittest.TestCase):
    def test_only_affordable_cards_are_legal(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=11)
        legal = env.legal_actions(state)
        self.assertEqual(legal[-1], END_TURN)
        for action in legal[:-1]:
            self.assertLessEqual(state.me.hand[action.index].cost, state.me.mana)

    def test_terminal_state_has_no_actions(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=5)
        finished = state.__class__(
            me=state.me,
            opponent=state.opponent,
            turn=state.turn,
            done=True,
            winner=0,
        )
        self.assertEqual(env.legal_actions(finished), ())


class OrderSensitivityTests(unittest.TestCase):
    """The headline property: playing Focus before attackers beats after."""

    def _hand_state(self, hand: tuple[int, ...], mana: int) -> GameState:
        env = ToyCardBattle()
        base = env.reset(seed=0)
        cards = tuple(CARD_LIBRARY[card_id] for card_id in hand)
        side = base.me.__class__(
            hero_hp=BASE_HP,
            mana=mana,
            hand=cards,
            deck=(),
            field=(),
        )
        opponent = base.opponent.__class__(
            hero_hp=BASE_HP,
            mana=0,
            hand=(),
            deck=(),
            field=(),
        )
        return GameState(me=side, opponent=opponent, turn=1)

    def _opponent_hp_after(self, order: tuple[int, ...], mana: int) -> int:
        env = ToyCardBattle()
        state = self._hand_state(order, mana)
        for card_id in order:
            index = next(
                i for i, card in enumerate(state.me.hand) if card.card_id == card_id
            )
            state = env.step(state, Action(kind="play", index=index))
        return state.opponent.hero_hp

    def test_focus_first_beats_focus_last(self) -> None:
        # Focus (3) then two Strikes (0) vs the reverse order.
        focus_first = self._opponent_hp_after((3, 0, 0), mana=3)
        focus_last = self._opponent_hp_after((0, 0, 3), mana=3)
        self.assertLess(focus_first, focus_last)


class GameLoopTests(unittest.TestCase):
    def test_random_playout_terminates(self) -> None:
        env = ToyCardBattle()
        for seed in range(25):
            state = env.reset(seed=seed)
            steps = 0
            while not state.done:
                legal = env.legal_actions(state)
                self.assertTrue(legal, "non-terminal state must expose an action")
                state = env.step(state, legal[0] if steps % 3 else END_TURN)
                steps += 1
                self.assertLess(steps, 500, "playout failed to terminate")
            self.assertIn(state.winner, (0, 1, 2))

    def test_hand_cap_respected(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=4)
        for _ in range(MAX_TURNS):
            if state.done:
                break
            state = env.step(state, END_TURN)
            self.assertLessEqual(len(state.me.hand), MAX_HAND)

    def test_step_on_terminal_is_identity(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=2)
        while not state.done:
            state = env.step(state, END_TURN)
        self.assertEqual(env.step(state, END_TURN), state)

    def test_minions_deal_permanent_damage(self) -> None:
        env = ToyCardBattle()
        state = env.reset(seed=0)
        card = CARD_LIBRARY[6]  # Golem: 2 attack
        side = state.me.__class__(
            hero_hp=BASE_HP,
            mana=card.cost,
            hand=(card,),
            deck=(),
            field=(),
        )
        opponent = state.opponent.__class__(
            hero_hp=BASE_HP, mana=0, hand=(), deck=(), field=()
        )
        state = GameState(me=side, opponent=opponent, turn=1)
        state = env.step(state, Action(kind="play", index=0))
        before = state.opponent.hero_hp
        state = env.step(state, END_TURN)
        # The minion hit for 2, then the opponent took its own greedy turn.
        self.assertLess(state.opponent.hero_hp, before)


if __name__ == "__main__":
    unittest.main()
