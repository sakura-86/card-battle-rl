"""card-battle-rl: reproducible turn-based card battle experiments.

Public surface is intentionally small; see ``README.md`` for the experiment
narrative and ``card_battle_rl.env`` for the environment contract.
"""

from .env import END_TURN, Action, Card, CardKind, GameState, Minion, Side, ToyCardBattle

__version__ = "0.1.0"

__all__ = [
    "END_TURN",
    "Action",
    "Card",
    "CardKind",
    "GameState",
    "Minion",
    "Side",
    "ToyCardBattle",
    "__version__",
]
