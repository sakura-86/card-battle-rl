"""PPO with GAE, written to be read.

Only the parts that matter for this experiment are here: clipped surrogate
objective, generalised advantage estimation, and a value loss.  No distributed
machinery, no framework dependencies beyond PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .features import ACTION_SPACE, CARD_DIM, GLOBAL_DIM, MAX_SLOTS
from .policy import PolicyValueNet


@dataclass
class Rollout:
    """A flat buffer of transitions from one or more episodes."""

    global_feats: torch.Tensor  # (N, G)
    card_feats: torch.Tensor  # (N, S, C)
    masks: torch.Tensor  # (N, A) bool
    actions: torch.Tensor  # (N,) long
    log_probs: torch.Tensor  # (N,)
    values: torch.Tensor  # (N,)
    rewards: torch.Tensor  # (N,)
    dones: torch.Tensor  # (N,) float

    def __len__(self) -> int:
        return int(self.actions.shape[0])


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """Generalised advantage estimation over a flat, chronological buffer.

    ``dones[t] == 1`` marks the last step of an episode, so the bootstrap term is
    dropped there.
    """
    advantages = torch.zeros_like(rewards)
    running = 0.0
    for step in reversed(range(len(rewards))):
        if step + 1 < len(rewards):
            next_value = values[step + 1]
            next_non_terminal = 1.0 - dones[step]
        else:
            next_value = torch.zeros((), dtype=values.dtype)
            next_non_terminal = 0.0
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        running = delta + gamma * lam * next_non_terminal * running
        advantages[step] = running
    return advantages


def ppo_update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    *,
    epochs: int = 4,
    batch_size: int = 256,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.95,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run ``epochs`` passes of clipped PPO over ``rollout``."""
    advantages = compute_gae(
        rollout.rewards, rollout.values, rollout.dones, gamma=gamma, lam=lam
    )
    returns = advantages + rollout.values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total = len(rollout)
    indices = torch.arange(total)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}

    for _ in range(epochs):
        permutation = indices[torch.randperm(total)]
        for start in range(0, total, batch_size):
            batch = permutation[start : start + batch_size]
            logits, values = net(
                rollout.global_feats[batch],
                rollout.card_feats[batch],
                rollout.masks[batch],
            )
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(rollout.actions[batch])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - rollout.log_probs[batch])
            unclipped = ratio * advantages[batch]
            clipped = (
                torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                * advantages[batch]
            )
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (values - returns[batch]).pow(2).mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["updates"] += 1

    if stats["updates"]:
        for key in ("policy_loss", "value_loss", "entropy"):
            stats[key] /= stats["updates"]
    return stats
