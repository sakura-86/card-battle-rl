"""A small deterministic turn-based card battle.

Design goals
------------
1. **Order sensitivity.** ``Focus`` grants a per-turn attack bonus that only
   applies to attacks played *after* it.  The same hand therefore has very
   different value depending on ordering, which is what makes whole-turn
   planning meaningfully different from step-by-step search.
2. **Tiny state.** No combat resolution, no damage assignment, no hidden zones
   beyond the shuffle order.  A full turn fits comfortably inside a search tree.
3. **Deterministic opponent.** The opponent is a pure function of the state, so
   two visits to the same node always agree.  Any stochasticity lives in the
   shuffle, which happens once at ``reset``.

Ruleset
-------
* Both heroes start at 30 HP.
* Each turn the active side gains ``min(turn, MAX_MANA)`` mana and draws one
  card.  Hand size is capped at ``MAX_HAND``; excess draws are discarded.
* Playing a card spends its cost.  ``end_turn`` resolves permanent attacks, then
  hands the turn to the opponent, who plays its entire turn before control
  returns.
* Reaching zero HP loses.  If ``MAX_TURNS`` passes with both alive, the higher
  hero HP wins.
* Drawing from an empty deck is a no-op (no fatigue damage), keeping games
  bounded without an extra counter.
"""

from __future__ import annotations

import random
from dataclasses import replace

from .core import (
    END_TURN,
    Action,
    Card,
    CardKind,
    GameState,
    Minion,
    Side,
    swap_perspective,
)

BASE_HP = 30
MAX_MANA = 6
MAX_HAND = 7
MAX_TURNS = 20

CARD_LIBRARY: tuple[Card, ...] = (
    # Attacks are deliberately *weak per card*.  Raw damage alone cannot close a
    # game, which is what makes the buff worth spending a turn on.
    Card(card_id=0, name="Strike", cost=1, kind=CardKind.ATTACK, power=1),
    Card(card_id=1, name="Bolt", cost=2, kind=CardKind.ATTACK, power=2),
    Card(card_id=2, name="Heavy Strike", cost=3, kind=CardKind.ATTACK, power=4),
    # One mana for +3 to every later attack: pays for itself at two attackers.
    Card(card_id=3, name="Focus", cost=1, kind=CardKind.BUFF, power=3),
    Card(card_id=4, name="Mend", cost=2, kind=CardKind.HEAL, power=6),
    Card(card_id=5, name="Insight", cost=1, kind=CardKind.DRAW, power=2),
    Card(card_id=6, name="Golem", cost=2, kind=CardKind.MINION, power=1, health=3),
    Card(card_id=7, name="Titan", cost=4, kind=CardKind.MINION, power=2, health=5),
)

# (card_id, copies); total == 14 cards per deck.
DECK_RECIPE: tuple[tuple[int, int], ...] = (
    (0, 4),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
)

PLANNER_CARD_ORDER = tuple(card.card_id for card in CARD_LIBRARY)


def build_deck(rng: random.Random) -> tuple[Card, ...]:
    """Materialise one shuffled deck from ``DECK_RECIPE``."""
    cards: list[Card] = []
    for card_id, copies in DECK_RECIPE:
        cards.extend([CARD_LIBRARY[card_id]] * copies)
    rng.shuffle(cards)
    return tuple(cards)


def _draw(side: Side, count: int) -> Side:
    """Draw up to ``count`` cards, honouring the hand cap."""
    drawn = side.deck[:count]
    rest = side.deck[count:]
    hand = side.hand + drawn
    if len(hand) > MAX_HAND:
        hand = hand[-MAX_HAND:]  # oldest cards are burned
    return replace(side, hand=hand, deck=rest)


def _resolve_card(active: Side, passive: Side, card: Card) -> tuple[Side, Side]:
    """Apply ``card`` played by ``active`` against ``passive``."""
    bonus = active.attack_bonus
    if card.kind is CardKind.ATTACK:
        inflicted = card.power + bonus
        return active, replace(passive, hero_hp=passive.hero_hp - inflicted)
    if card.kind is CardKind.HEAL:
        healed = min(BASE_HP, active.hero_hp + card.power)
        return replace(active, hero_hp=healed), passive
    if card.kind is CardKind.DRAW:
        return _draw(active, card.power), passive
    if card.kind is CardKind.BUFF:
        return replace(active, attack_bonus=bonus + card.power), passive
    if card.kind is CardKind.MINION:
        minion = Minion(attack=card.power, health=card.health)
        return replace(active, field=active.field + (minion,)), passive
    raise ValueError(f"unsupported card kind: {card.kind!r}")


def _greedy_score(active: Side, passive: Side, card: Card) -> float:
    """Heuristic value of a card from ``active``'s perspective.

    Deliberately simple: the opponent is a baseline, not a strong player.
    """
    bonus = active.attack_bonus
    if card.kind is CardKind.ATTACK:
        return float(card.power + bonus)
    if card.kind is CardKind.BUFF:
        attackers = sum(1 for c in active.hand if c.kind is CardKind.ATTACK)
        return float(card.power * min(attackers, 3))
    if card.kind is CardKind.MINION:
        return float(card.power * 3)
    if card.kind is CardKind.DRAW:
        return float(card.power * 1.5)
    if card.kind is CardKind.HEAL:
        missing = BASE_HP - active.hero_hp
        return float(min(card.power, missing))
    return 0.0


class ToyCardBattle:
    """Concrete environment implementing :class:`card_battle_rl.env.Environment`.

    ``opponent`` is an optional agent object exposing ``choose(env, state)``.
    When omitted the built-in greedy policy drives the second seat, which keeps
    search code free of opponent recursion.
    """

    def __init__(self, *, max_turns: int = MAX_TURNS, opponent: object | None = None) -> None:
        self.max_turns = max_turns
        self.opponent_policy = opponent

    # -- lifecycle ---------------------------------------------------------

    def reset(self, seed: int) -> GameState:
        deal_rng = random.Random(seed)
        my_deck = build_deck(deal_rng)
        opp_deck = build_deck(deal_rng)

        me = Side(hero_hp=BASE_HP, mana=1, deck=my_deck)
        opponent = Side(hero_hp=BASE_HP, mana=0, deck=opp_deck)
        me = _draw(me, 4)
        opponent = _draw(opponent, 4)
        return GameState(me=me, opponent=opponent, turn=1)

    # -- queries -----------------------------------------------------------

    def is_terminal(self, state: GameState) -> bool:
        return state.done or state.turn_closed

    def legal_actions(self, state: GameState) -> tuple[Action, ...]:
        if state.done or state.turn_closed:
            return ()
        playable = [
            Action(kind="play", index=index)
            for index, card in enumerate(state.me.hand)
            if card.cost <= state.me.mana
        ]
        playable.append(END_TURN)
        return tuple(playable)

    def reward(self, state: GameState) -> float:
        """Terminal reward from the perspective of ``state.me``."""
        if not state.done:
            return 0.0
        if state.winner == 0:
            return 1.0
        if state.winner == 1:
            return -1.0
        return 0.0

    # -- transitions -------------------------------------------------------

    def step(self, state: GameState, action: Action) -> GameState:
        if state.done or state.turn_closed:
            return state
        if action.kind == "end_turn":
            return self._end_turn(state)
        if action.kind != "play":
            raise ValueError(f"unknown action kind: {action.kind!r}")
        return self._play(state, action.index)

    def _play(self, state: GameState, index: int) -> GameState:
        card = state.me.hand[index]
        if card.cost > state.me.mana:
            raise ValueError(f"card {card.name!r} is not affordable")
        me = replace(
            state.me,
            mana=state.me.mana - card.cost,
            hand=state.me.hand[:index] + state.me.hand[index + 1 :],
        )
        me, opponent = _resolve_card(me, state.opponent, card)
        return self._check_terminal(replace(state, me=me, opponent=opponent))

    def resolve_turn(self, state: GameState) -> GameState:
        """Resolve end-of-turn effects for the active side only.

        Permanent damage lands, the attack bonus resets, and the resulting state
        is flagged ``turn_closed`` so search stops expanding there.  The opponent
        never gets simulated, which keeps a search tree inside one turn.
        """
        permanent_damage = sum(minion.attack for minion in state.me.field)
        opponent = replace(state.opponent, hero_hp=state.opponent.hero_hp - permanent_damage)
        me = replace(state.me, attack_bonus=0)
        resolved = replace(state, me=me, opponent=opponent, turn_closed=True)
        return self._check_terminal(resolved)

    def _end_turn(self, state: GameState) -> GameState:
        state = self.resolve_turn(state)
        if state.done:
            return state
        state = replace(state, turn_closed=False)

        state = self._opponent_turn(state)
        if state.done:
            return state

        next_turn = state.turn + 1
        me = _draw(state.me, 1)
        me = replace(me, mana=min(next_turn, MAX_MANA))
        state = replace(state, me=me, turn=next_turn)
        if next_turn > self.max_turns:
            return self._decide_on_hp(state)
        return state

    def _opponent_turn(self, state: GameState) -> GameState:
        opponent = _draw(state.opponent, 1)
        opponent = replace(
            opponent, mana=min(state.turn, MAX_MANA), attack_bonus=0
        )
        state = replace(state, opponent=opponent)

        if self.opponent_policy is not None:
            state = self._driven_opponent_turn(state)
        else:
            state = self._greedy_opponent_turn(state)

        if state.done:
            return state

        opponent, me = state.opponent, state.me
        permanent_damage = sum(minion.attack for minion in opponent.field)
        me = replace(me, hero_hp=me.hero_hp - permanent_damage)
        opponent = replace(opponent, attack_bonus=0)
        return self._check_terminal(replace(state, me=me, opponent=opponent))

    def _greedy_opponent_turn(self, state: GameState) -> GameState:
        while not state.done:
            opponent, me = state.opponent, state.me
            affordable = [
                (index, card)
                for index, card in enumerate(opponent.hand)
                if card.cost <= opponent.mana
            ]
            if not affordable:
                break
            index, card = max(
                affordable, key=lambda pair: (_greedy_score(opponent, me, pair[1]), -pair[0])
            )
            opponent = replace(
                opponent,
                mana=opponent.mana - card.cost,
                hand=opponent.hand[:index] + opponent.hand[index + 1 :],
            )
            opponent, me = _resolve_card(opponent, me, card)
            state = self._check_terminal(replace(state, me=me, opponent=opponent))
        return state

    def _driven_opponent_turn(self, state: GameState) -> GameState:
        """Let ``opponent_policy`` play the whole responding turn."""
        max_plays = len(state.opponent.hand) + 1
        for _ in range(max_plays):
            if state.done:
                return state
            action = self.opponent_policy.choose(self, swap_perspective(state))  # type: ignore[union-attr]
            if action.kind == "end_turn":
                return state
            card = state.opponent.hand[action.index]
            if card.cost > state.opponent.mana:
                return state  # policy returned an illegal play; stop the turn
            opponent = replace(
                state.opponent,
                mana=state.opponent.mana - card.cost,
                hand=state.opponent.hand[: action.index]
                + state.opponent.hand[action.index + 1 :],
            )
            opponent, me = _resolve_card(opponent, state.me, card)
            state = self._check_terminal(
                replace(state, me=me, opponent=opponent)
            )
        return state

    # -- helpers -----------------------------------------------------------

    def _check_terminal(self, state: GameState) -> GameState:
        me_dead = state.me.hero_hp <= 0
        opponent_dead = state.opponent.hero_hp <= 0
        if me_dead and opponent_dead:
            return replace(state, done=True, winner=2)
        if opponent_dead:
            return replace(state, done=True, winner=0)
        if me_dead:
            return replace(state, done=True, winner=1)
        return state

    def _decide_on_hp(self, state: GameState) -> GameState:
        my_hp, opp_hp = state.me.hero_hp, state.opponent.hero_hp
        if my_hp > opp_hp:
            winner = 0
        elif opp_hp > my_hp:
            winner = 1
        else:
            winner = 2
        return replace(state, done=True, winner=winner)
