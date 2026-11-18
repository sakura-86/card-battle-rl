"""Fixed-size encodings of the toy state for the learning agent.

Two design rules, both learned the hard way in the larger project:

1. **No absolute identifiers.**  Cards are encoded by their *properties*
   (cost, power, kind), never by ``card_id``.  A network that memorises
   ``card_id == 3`` cannot generalise to a card it has not seen.
2. **No leakage.**  Nothing from the future (draw order, opponent hand contents)
   enters the feature vector.  Only what a player could legally observe.

The action space is a pointer over hand slots plus one ``end_turn`` slot, which
keeps the head size fixed no matter how the hand changes.
"""

from __future__ import annotations

import numpy as np

from ..env.core import CardKind, GameState
from ..env.toy import BASE_HP, MAX_MANA
from ..search.evaluation import _affordable_attack_damage

MAX_SLOTS = 7
GLOBAL_DIM = 12
CARD_DIM = 12
ACTION_SPACE = MAX_SLOTS + 1
END_TURN_INDEX = MAX_SLOTS

MAX_POWER = 10.0
MAX_BONUS = 6.0
MAX_TURNS = 20.0
MAX_FIELD_ATTACK = 8.0
MAX_POTENTIAL = 20.0

_KIND_ORDER = (
    CardKind.ATTACK,
    CardKind.HEAL,
    CardKind.DRAW,
    CardKind.BUFF,
    CardKind.MINION,
)


def _kind_onehot(kind: CardKind) -> list[float]:
    return [1.0 if kind is candidate else 0.0 for candidate in _KIND_ORDER]


def encode_global(state: GameState) -> np.ndarray:
    """Aggregate, side-relative features."""
    affordable = [card for card in state.me.hand if card.cost <= state.me.mana]
    features = [
        state.me.hero_hp / BASE_HP,
        state.opponent.hero_hp / BASE_HP,
        state.me.mana / MAX_MANA,
        state.me.attack_bonus / MAX_BONUS,
        state.turn / MAX_TURNS,
        sum(minion.attack for minion in state.me.field) / MAX_FIELD_ATTACK,
        sum(minion.attack for minion in state.opponent.field) / MAX_FIELD_ATTACK,
        len(state.me.hand) / MAX_SLOTS,
        len(state.opponent.hand) / MAX_SLOTS,
        len(affordable) / MAX_SLOTS,
        _affordable_attack_damage(state.me, state.turn_closed) / MAX_POTENTIAL,
        1.0,
    ]
    return np.asarray(features, dtype=np.float32)


def encode_cards(state: GameState) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot card features plus the legal-action mask.

    Returns ``(cards, mask)`` where ``cards`` has shape ``(MAX_SLOTS, CARD_DIM)``
    and ``mask`` has shape ``(ACTION_SPACE,)``.
    """
    cards = np.zeros((MAX_SLOTS, CARD_DIM), dtype=np.float32)
    mask = np.zeros(ACTION_SPACE, dtype=bool)

    for slot, card in enumerate(state.me.hand[:MAX_SLOTS]):
        affordable = 1.0 if card.cost <= state.me.mana else 0.0
        features = [
            card.cost / MAX_MANA,
            card.power / MAX_POWER,
            card.health / MAX_POWER,
            *_kind_onehot(card.kind),
            affordable,
            1.0 if card.kind is CardKind.ATTACK else 0.0,
            1.0 if card.kind is CardKind.BUFF else 0.0,
            1.0,
        ]
        cards[slot] = np.asarray(features, dtype=np.float32)
        if affordable:
            mask[slot] = True

    mask[END_TURN_INDEX] = True
    return cards, mask


def encode_action_index(action_kind: str, action_index: int) -> int:
    return END_TURN_INDEX if action_kind == "end_turn" else action_index


def decode_action_index(index: int) -> tuple[str, int]:
    return ("end_turn", -1) if index == END_TURN_INDEX else ("play", index)
