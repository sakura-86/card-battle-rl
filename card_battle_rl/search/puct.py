"""A compact PUCT search over the toy environment.

This is the same family of algorithm as the original AlphaZero search, stripped
of everything game-specific: a tree of states, a uniform prior when no neural
network is attached, and a value read straight off the leaf.

Two knobs matter for the experiments:

``simulations``
    How many leaf evaluations a single decision may spend.  This is the direct
    analogue of "N simulations" in AlphaZero-style agents.

``c_puct``
    Exploration constant in the selection formula::

        Q + c_puct * P * sqrt(sum(N)) / (1 + N)

Because the prior is uniform, ``P`` is ``1/len(actions)`` everywhere.  That is
the deliberate baseline: it isolates the *effect of search*, with no learned
knowledge to help.
"""

from __future__ import annotations

import math

from ..env.core import END_TURN, Action, GameState
from ..env.toy import ToyCardBattle
from .evaluation import heuristic_value

DEFAULT_SIMULATIONS = 64
# Small budgets are the regime this repository cares about.  With a uniform prior
# and c_puct much above 0.5 the exploration term swamps the value differences a
# heuristic evaluator can produce, visits flatten out, and the agent degenerates
# into a random picker.
DEFAULT_C_PUCT = 0.5


class _Node:
    """One search node.  Immutable state, mutable visit statistics."""

    __slots__ = ("state", "prior", "visits", "value_sum", "children")

    def __init__(self, state: GameState, prior: float = 0.0) -> None:
        self.state = state
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[Action, _Node] = {}

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children) or self.state.done or self.state.turn_closed


def _expand(env: ToyCardBattle, node: _Node) -> None:
    """Materialise every child of ``node`` with a uniform prior.

    ``end_turn`` is resolved with :meth:`ToyCardBattle.resolve_turn` rather than
    a full ``step``.  Advancing the whole opponent turn here would let a search
    node recursively invoke its own agent, which is how a 32-simulation search
    turns into unbounded recursion.
    """
    actions = env.legal_actions(node.state)
    if not actions:
        return
    prior = 1.0 / len(actions)
    children: dict[Action, _Node] = {}
    for action in actions:
        if action.kind == "end_turn":
            child_state = env.resolve_turn(node.state)
        else:
            child_state = env.step(node.state, action)
        children[action] = _Node(child_state, prior=prior)
    node.children = children


def _select_child(node: _Node, c_puct: float) -> _Node:
    """UCB1-tuned selection with a prior term."""
    sqrt_parent = math.sqrt(node.visits)
    best_child: _Node | None = None
    best_score = float("-inf")
    for child in node.children.values():
        exploitation = child.mean_value
        exploration = c_puct * child.prior * sqrt_parent / (1 + child.visits)
        score = exploitation + exploration
        if score > best_score:
            best_score = score
            best_child = child
    assert best_child is not None, "selection requires at least one child"
    return best_child


def _backup(path: list[_Node], value: float) -> None:
    for node in path:
        node.visits += 1
        node.value_sum += value


def search(
    env: ToyCardBattle,
    state: GameState,
    *,
    simulations: int = DEFAULT_SIMULATIONS,
    c_puct: float = DEFAULT_C_PUCT,
) -> _Node:
    """Run ``simulations`` leaf evaluations and return the populated root."""
    root = _Node(state)
    for _ in range(simulations):
        path = [root]
        node = root
        while node.children and not node.state.done:
            node = _select_child(node, c_puct)
            path.append(node)
        if not node.expanded:
            _expand(env, node)
        _backup(path, heuristic_value(node.state))
    return root


def policy_target(root: _Node) -> dict[Action, float]:
    """Visit-count distribution, suitable as a policy learning target."""
    total = sum(child.visits for child in root.children.values())
    if total == 0:
        uniform = 1.0 / len(root.children) if root.children else 0.0
        return {action: uniform for action in root.children}
    return {action: child.visits / total for action, child in root.children.items()}


class PUCTAgent:
    """Agent wrapper exposing the same ``choose`` interface as the baselines."""

    def __init__(
        self,
        simulations: int = DEFAULT_SIMULATIONS,
        c_puct: float = DEFAULT_C_PUCT,
        name: str | None = None,
        select_by: str = "visits",
    ) -> None:
        if select_by not in ("visits", "q"):
            raise ValueError(f"select_by must be 'visits' or 'q', got {select_by!r}")
        self.simulations = simulations
        self.c_puct = c_puct
        self.select_by = select_by
        suffix = "" if select_by == "visits" else "_q"
        self.name = name or f"puct_{simulations}{suffix}"

    def root(self, env: ToyCardBattle, state: GameState) -> _Node:
        return search(
            env, state, simulations=self.simulations, c_puct=self.c_puct
        )

    def choose(self, env: ToyCardBattle, state: GameState) -> Action:
        root = self.root(env, state)
        if not root.children:
            return END_TURN
        if self.select_by == "q":
            # With few simulations, visits are a noisy proxy; mean value is the
            # more direct read on which move looked best.
            return max(
                root.children,
                key=lambda action: (
                    root.children[action].mean_value,
                    root.children[action].visits,
                ),
            )
        return max(root.children, key=lambda action: root.children[action].visits)
