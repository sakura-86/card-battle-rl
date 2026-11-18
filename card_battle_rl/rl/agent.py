"""Agent wrapper around the trained network."""

from __future__ import annotations

import torch

from ..env.core import Action, GameState
from ..env.toy import ToyCardBattle
from .features import decode_action_index, encode_cards, encode_global
from .policy import PolicyValueNet


class RLAgent:
    """Greedy policy from a trained :class:`PolicyValueNet`."""

    def __init__(self, net: PolicyValueNet, name: str = "ppo") -> None:
        self.net = net
        self.name = name

    def choose(self, env: ToyCardBattle, state: GameState) -> Action:
        global_tensor = torch.from_numpy(encode_global(state)).unsqueeze(0)
        cards, mask = encode_cards(state)
        cards_tensor = torch.from_numpy(cards).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(global_tensor, cards_tensor, mask_tensor)
        index = int(torch.argmax(logits, dim=-1))
        kind, card_index = decode_action_index(index)
        return Action(kind=kind, index=card_index)
