"""Tests for the learning stack: features, network shapes, GAE."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from card_battle_rl.env import GameState, ToyCardBattle
from card_battle_rl.env.toy import BASE_HP, CARD_LIBRARY
from card_battle_rl.rl.features import (
    ACTION_SPACE,
    CARD_DIM,
    END_TURN_INDEX,
    GLOBAL_DIM,
    MAX_SLOTS,
    decode_action_index,
    encode_action_index,
    encode_cards,
    encode_global,
)
from card_battle_rl.rl.policy import PolicyValueNet
from card_battle_rl.rl.ppo import compute_gae


def build_state(hand: tuple[int, ...], mana: int) -> GameState:
    env = ToyCardBattle()
    base = env.reset(seed=0)
    cards = tuple(CARD_LIBRARY[card_id] for card_id in hand)
    me = base.me.__class__(hero_hp=BASE_HP, mana=mana, hand=cards)
    opponent = base.opponent.__class__(hero_hp=BASE_HP, mana=0)
    return GameState(me=me, opponent=opponent, turn=1)


class FeatureShapeTests(unittest.TestCase):
    def test_shapes(self) -> None:
        state = build_state((0, 1, 3), mana=4)
        self.assertEqual(encode_global(state).shape, (GLOBAL_DIM,))
        cards, mask = encode_cards(state)
        self.assertEqual(cards.shape, (MAX_SLOTS, CARD_DIM))
        self.assertEqual(mask.shape, (ACTION_SPACE,))

    def test_mask_allows_only_affordable_cards(self) -> None:
        state = build_state((0, 2, 7), mana=3)  # Strike(1), Heavy(3), Titan(4)
        _, mask = encode_cards(state)
        self.assertTrue(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2], "Titan costs 4 with only 3 mana")
        self.assertTrue(mask[END_TURN_INDEX])

    def test_action_roundtrip(self) -> None:
        self.assertEqual(encode_action_index("end_turn", -1), END_TURN_INDEX)
        self.assertEqual(encode_action_index("play", 3), 3)
        self.assertEqual(decode_action_index(END_TURN_INDEX), ("end_turn", -1))
        self.assertEqual(decode_action_index(2), ("play", 2))

    def test_features_do_not_encode_card_identity(self) -> None:
        """Two Stacks of the same cost/power must produce identical features.

        Encoding ``card_id`` would let the network memorise specific cards instead
        of learning their properties.
        """
        first = build_state((0,), mana=1)
        second = build_state((0,), mana=1)
        self.assertTrue(
            np.array_equal(encode_cards(first)[0], encode_cards(second)[0])
        )

    def test_features_stay_finite(self) -> None:
        state = build_state((0, 3, 7), mana=6)
        self.assertTrue(np.all(np.isfinite(encode_global(state))))
        self.assertTrue(np.all(np.isfinite(encode_cards(state)[0])))


class NetworkTests(unittest.TestCase):
    def test_forward_shapes_and_masking(self) -> None:
        net = PolicyValueNet(hidden=16)
        batch = 4
        global_feat = torch.randn(batch, GLOBAL_DIM)
        card_feats = torch.randn(batch, MAX_SLOTS, CARD_DIM)
        mask = torch.zeros(batch, ACTION_SPACE, dtype=torch.bool)
        mask[:, END_TURN_INDEX] = True

        logits, value = net(global_feat, card_feats, mask)
        self.assertEqual(logits.shape, (batch, ACTION_SPACE))
        self.assertEqual(value.shape, (batch,))
        # Illegals must be pushed far below any legal option.
        illegal = logits[:, :MAX_SLOTS]
        self.assertTrue(torch.all(illegal < -1e8))

    def test_gradient_flows(self) -> None:
        net = PolicyValueNet(hidden=16)
        global_feat = torch.randn(2, GLOBAL_DIM)
        card_feats = torch.randn(2, MAX_SLOTS, CARD_DIM)
        mask = torch.ones(2, ACTION_SPACE, dtype=torch.bool)
        logits, value = net(global_feat, card_feats, mask)
        loss = logits.sum() + value.sum()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.all(torch.isfinite(g)) for g in grads))


class GAETests(unittest.TestCase):
    def test_terminal_step_does_not_bootstrap(self) -> None:
        rewards = torch.tensor([0.0, 1.0])
        values = torch.tensor([0.5, 0.5])
        dones = torch.tensor([0.0, 1.0])
        advantages = compute_gae(rewards, values, dones, gamma=0.9, lam=1.0)
        # Second step: r=1, no bootstrap -> advantage = 1 - 0.5 = 0.5
        self.assertAlmostEqual(float(advantages[1]), 0.5, places=5)

    def test_non_terminal_bootstraps(self) -> None:
        rewards = torch.tensor([0.0, 0.0])
        values = torch.tensor([0.0, 0.0])
        dones = torch.tensor([0.0, 0.0])
        advantages = compute_gae(rewards, values, dones, gamma=1.0, lam=1.0)
        self.assertAlmostEqual(float(advantages[0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
