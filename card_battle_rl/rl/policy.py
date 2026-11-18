"""Policy/value network for the learning agent.

The head is a *pointer over hand slots* rather than a fixed vocabulary of card
ids.  That choice is what lets one network handle any hand composition: each slot
is scored by pairing its own features with a shared state embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .features import ACTION_SPACE, CARD_DIM, GLOBAL_DIM, MAX_SLOTS


class PolicyValueNet(nn.Module):
    """Small pointer head with a scalar value output."""

    def __init__(
        self,
        global_dim: int = GLOBAL_DIM,
        card_dim: int = CARD_DIM,
        hidden: int = 64,
        slots: int = MAX_SLOTS,
    ) -> None:
        super().__init__()
        self.slots = slots
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU()
        )
        self.card_encoder = nn.Sequential(nn.Linear(card_dim, hidden), nn.ReLU())
        self.card_scorer = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.end_turn_scorer = nn.Linear(hidden, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        global_feat: torch.Tensor,
        card_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, value)``.

        ``global_feat``: ``(B, G)``; ``card_feats``: ``(B, S, C)``;
        ``mask``: ``(B, A)`` boolean, ``True`` for legal actions.
        """
        state_embedding = self.global_encoder(global_feat)
        card_embeddings = self.card_encoder(card_feats)
        expanded = state_embedding.unsqueeze(1).expand(-1, self.slots, -1)
        pair = torch.cat([expanded, card_embeddings], dim=-1)
        card_logits = self.card_scorer(pair).squeeze(-1)
        end_turn_logits = self.end_turn_scorer(state_embedding)
        logits = torch.cat([card_logits, end_turn_logits], dim=-1)
        # A large negative constant rather than -inf: -inf produces NaNs once
        # softmax gradients flow through an all-masked row.
        logits = logits.masked_fill(~mask, -1e9)
        value = self.value_head(state_embedding).squeeze(-1)
        return logits, value
