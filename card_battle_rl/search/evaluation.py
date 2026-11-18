"""Hand-tuned position evaluation shared by every agent.

Using one evaluator for all agents keeps the experiment honest: the only thing
that varies between ``GreedyBot``, ``RulePlanner`` and ``PUCT`` is *how they
search*, never *how they read a position*.

The evaluator scores a single quantity -- the **expected final hero HP gap**::

    (my_hp + damage_still_available - projected_opponent_hp) / BASE_HP

Three consequences, each of which was a bug at some point:

* **Bookkeeping is conserved.**  Dealing one point of damage moves it out of
  ``damage_still_available`` and into ``opponent_hp``, so spending damage now is
  neutral rather than penalised.  Scoring HP and potential with different
  denominators made "play a card" look worse than "do nothing".
* **Signs are consistent.**  More own HP and less opponent HP is better.  The
  mirrored phrasing taught a search agent to avoid dealing damage and lost 40
  games in a row.
* **A closed turn keeps no potential.**  ``resolve_turn`` does not zero leftover
  mana, so without an explicit guard an ended turn still looked like it had
  damage in hand.

What this evaluator deliberately *cannot* do is price ordering.  Goldfish-style
potential ignores that ``Focus`` must precede the attacks it amplifies; that
combination only appears once search actually sequences the cards.  The tension is
the subject of the experiments.
"""

from __future__ import annotations

from ..env.core import CardKind, GameState, Side
from ..env.toy import BASE_HP

MAX_FIELD_ATTACK = 8.0

W_HP = 0.85
W_FIELD = 0.15


def _projected_opponent_hp(state: GameState) -> int:
    """Opponent HP after permanent damage that is already locked in.

    ``resolve_turn`` has applied it for closed turns; for open turns it is still
    pending.  Projecting it onto both keeps leaf scores comparable.
    """
    hp = state.opponent.hero_hp
    if not state.turn_closed:
        hp -= sum(minion.attack for minion in state.me.field)
    return hp


def _affordable_attack_damage(side: Side, turn_closed: bool) -> int:
    """Damage the side can still deal this turn, greedily packed by value per mana.

    A closed turn cannot play anything else, no matter how much mana is left over.
    """
    if turn_closed:
        return 0
    attackers = [
        card
        for card in side.hand
        if card.kind is CardKind.ATTACK and card.cost <= side.mana
    ]
    attackers.sort(
        key=lambda card: (card.power + side.attack_bonus) / max(card.cost, 1),
        reverse=True,
    )
    budget = side.mana
    total = 0
    for card in attackers:
        if card.cost > budget:
            continue
        total += card.power + side.attack_bonus
        budget -= card.cost
    return total


def heuristic_value(state: GameState) -> float:
    """Value estimate for ``state`` from ``state.me``'s perspective."""
    if state.done:
        if state.winner == 0:
            return 1.0
        if state.winner == 1:
            return -1.0
        return 0.0

    potential = _affordable_attack_damage(state.me, state.turn_closed)
    expected_opponent_hp = _projected_opponent_hp(state) - potential
    hp_advantage = (state.me.hero_hp - expected_opponent_hp) / BASE_HP

    my_board = sum(minion.attack for minion in state.me.field)
    opp_board = sum(minion.attack for minion in state.opponent.field)
    board_advantage = (my_board - opp_board) / MAX_FIELD_ATTACK

    return W_HP * hp_advantage + W_FIELD * board_advantage
