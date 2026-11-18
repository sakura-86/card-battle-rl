"""Core value types for a turn-based card battle environment.

The state model here is deliberately minimal: a small immutable ``GameState``
plus a small immutable ``Action``.  Keeping both frozen means search code never
needs an undo log -- a tree node just holds a state reference, and ``step``
returns a brand new state.

Nothing in this module knows about any specific game.  ``env.toy`` supplies one
concrete ruleset; swapping it for another is the intended extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class CardKind(str, Enum):
    """Coarse card categories the toy ruleset understands."""

    ATTACK = "attack"
    HEAL = "heal"
    DRAW = "draw"
    BUFF = "buff"
    MINION = "minion"


@dataclass(frozen=True, slots=True)
class Card:
    """A single card definition.  Card instances are shared, never mutated."""

    card_id: int
    name: str
    cost: int
    kind: CardKind
    power: int = 0
    health: int = 0


@dataclass(frozen=True, slots=True)
class Minion:
    """A permanent on the board.

    The toy ruleset keeps combat out of scope: minions deal their attack to the
    opposing hero at end of turn and are never damaged.
    """

    attack: int
    health: int


@dataclass(frozen=True, slots=True)
class Side:
    """Everything belonging to one player.

    ``attack_bonus`` lives on the side because both players accumulate it during
    their own turn and it resets when that turn ends.
    """

    hero_hp: int
    mana: int = 0
    hand: tuple[Card, ...] = ()
    deck: tuple[Card, ...] = ()
    field: tuple[Minion, ...] = ()
    attack_bonus: int = 0


@dataclass(frozen=True, slots=True)
class GameState:
    """Full observable state of one game.

    ``turn_closed`` is a search-only flag: it marks that the active side has
    finished its turn but the opponent's turn has not been simulated.  Search
    code cuts the tree there so a single node never has to expand an entire
    opponent turn.
    """

    me: Side
    opponent: Side
    turn: int = 1
    done: bool = False
    winner: int = -1  # -1 nothing yet, 0 = me, 1 = opponent, 2 = draw
    turn_closed: bool = False


@dataclass(frozen=True, slots=True)
class Action:
    """A single decision point move.

    ``kind`` is ``"play"`` (play the card at ``index`` in hand) or
    ``"end_turn"``.
    """

    kind: str
    index: int = -1

    @property
    def is_terminal_action(self) -> bool:
        return self.kind == "end_turn"


END_TURN = Action(kind="end_turn")

_WINNER_SWAP = {-1: -1, 0: 1, 1: 0, 2: 2}


def swap_perspective(state: GameState) -> GameState:
    """Return ``state`` from the other player's point of view.

    Lets an agent that always plays "me" also drive the responding side without
    a second implementation.
    """
    return GameState(
        me=state.opponent,
        opponent=state.me,
        turn=state.turn,
        done=state.done,
        winner=_WINNER_SWAP[state.winner],
        turn_closed=state.turn_closed,
    )


class Environment(Protocol):
    """Minimal interface the search and RL code depends on."""

    def reset(self, seed: int) -> GameState: ...

    def legal_actions(self, state: GameState) -> tuple[Action, ...]: ...

    def step(self, state: GameState, action: Action) -> GameState: ...

    def is_terminal(self, state: GameState) -> bool: ...
