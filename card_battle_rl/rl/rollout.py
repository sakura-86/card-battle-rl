"""Rollout collection against a fixed opponent.

The learning agent always drives the first seat; the responding seat is supplied
by the environment.  Training against a *fixed* opponent keeps the objective
stationary, which is what makes a small PPO run reproducible.  Self-play is the
natural next step, not the starting point.
"""

from __future__ import annotations

import torch

from ..env.core import Action, GameState
from ..env.toy import BASE_HP, ToyCardBattle
from .features import decode_action_index, encode_cards, encode_global
from .policy import PolicyValueNet
from .ppo import Rollout

TERMINAL_REWARD = {0: 1.0, 1: -1.0, 2: 0.0}


def _shaping_reward(previous: GameState, current: GameState) -> float:
    """Dense signal: reward net hero damage traded in my favour.

    Without it a 20-turn game yields a single scalar and PPO has almost nothing
    to learn from early on.  The scale is kept small so it cannot dominate the
    terminal result.
    """
    damage_dealt = previous.opponent.hero_hp - current.opponent.hero_hp
    damage_taken = previous.me.hero_hp - current.me.hero_hp
    return (damage_dealt - damage_taken) / BASE_HP


def collect(
    env: ToyCardBattle,
    net: PolicyValueNet,
    *,
    episodes: int,
    seed_base: int,
    shaping_scale: float = 0.5,
) -> Rollout:
    global_feats: list[torch.Tensor] = []
    card_feats: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    actions: list[int] = []
    log_probs: list[float] = []
    values: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []

    net.eval()
    for episode in range(episodes):
        state = env.reset(seed_base + episode)
        while not state.done:
            encoded_global = encode_global(state)
            encoded_cards, mask = encode_cards(state)

            global_tensor = torch.from_numpy(encoded_global).unsqueeze(0)
            cards_tensor = torch.from_numpy(encoded_cards).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            with torch.no_grad():
                logits, value = net(global_tensor, cards_tensor, mask_tensor)
                distribution = torch.distributions.Categorical(logits=logits)
                chosen = distribution.sample()
                log_prob = distribution.log_prob(chosen)

            kind, index = decode_action_index(int(chosen))
            next_state = env.step(state, Action(kind=kind, index=index))

            reward = shaping_scale * _shaping_reward(state, next_state)
            if next_state.done:
                reward += TERMINAL_REWARD[next_state.winner]

            global_feats.append(global_tensor.squeeze(0))
            card_feats.append(cards_tensor.squeeze(0))
            masks.append(mask_tensor.squeeze(0))
            actions.append(int(chosen))
            log_probs.append(float(log_prob))
            values.append(float(value))
            rewards.append(reward)
            dones.append(1.0 if next_state.done else 0.0)

            state = next_state

    return Rollout(
        global_feats=torch.stack(global_feats),
        card_feats=torch.stack(card_feats),
        masks=torch.stack(masks),
        actions=torch.tensor(actions, dtype=torch.long),
        log_probs=torch.tensor(log_probs, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        rewards=torch.tensor(rewards, dtype=torch.float32),
        dones=torch.tensor(dones, dtype=torch.float32),
    )
