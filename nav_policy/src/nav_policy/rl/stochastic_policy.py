"""Gaussian actor + value critic for RL fine-tuning on BC/DAgger initializers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from nav_policy.data.normalization import CommandStats
from nav_policy.model.factory import build_model


class StochasticVelocityPolicy(nn.Module):
    """
    Wraps a deterministic BC policy with a diagonal Gaussian exploration head
    and a scalar value critic on the shared latent features.

    Actions are sampled in z-score space (same as BC training targets); the
    caller de-standardizes with ``CommandStats`` before sending to FiGS.
    """

    def __init__(self,
                 base: nn.Module,
                 init_log_std: float = -0.5,
                 critic_hidden: int = 256,
                 residual_actor: bool = False,
                 residual_scale: float = 1.0,
                 residual_hidden: int = 256) -> None:
        super().__init__()
        self.base = base
        self.cmd_dim = int(base.cmd_dim)
        latent_dim = int(base.gru_hidden) + int(base.goal_emb_dim)
        self.log_std = nn.Parameter(
            torch.full((self.cmd_dim,), float(init_log_std))
        )
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, critic_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(critic_hidden, 1),
        )
        self.use_depth = bool(getattr(base, "use_depth", False))

        # Residual actor head (for stable offline->online fine-tuning).
        # When enabled, the policy mean is the FROZEN BC mean plus a bounded
        # learnable correction:  mean = bc_mean + residual_scale * tanh(head(z)).
        # The head's final layer is zero-initialized so at step 0 the residual
        # is exactly 0 -> policy == BC seed (cannot regress below it), and the
        # tanh bounds the per-dim deviation to +/- residual_scale (in z-score
        # action units). See TD3+BC / residual-RL.
        self.residual_actor = bool(residual_actor)
        self.residual_scale = float(residual_scale)
        if self.residual_actor:
            self.actor_head = nn.Sequential(
                nn.Linear(latent_dim, residual_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(residual_hidden, self.cmd_dim),
            )
            # Zero-init the last layer -> residual starts at 0 (BC behavior).
            last = self.actor_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        else:
            self.actor_head = None

    def _encode(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single backbone pass; returns (latent [B, gru_hidden+goal_emb_dim],
        mean [B, cmd_dim]). Used by act/evaluate/kl_to/q_input so we don't
        call the backbone twice per call.
        """
        if self.use_depth:
            latent = self.base.forward_latent(rgb_seq, goal, depth_seq)
        else:
            latent = self.base.forward_latent(rgb_seq, goal)
        out = self.base.head(latent)
        bc_mean = out.view(-1, self.base.H, self.cmd_dim)[:, 0, :]
        if self.residual_actor:
            delta = self.residual_scale * torch.tanh(self.actor_head(latent))
            mean = bc_mean + delta
        else:
            mean = bc_mean
        return latent, mean

    def bc_mean(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The un-residualized BC mean (frozen base head, first horizon step).
        Used as the regression target for the TD3+BC action-space anchor.
        """
        return self._mean(rgb_seq, goal, depth_seq)

    def deterministic_mean(self,
                           rgb_seq: torch.Tensor,
                           goal: torch.Tensor,
                           depth_seq: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """(latent, policy mean) with the residual applied. Deterministic
        actor output for TD3-style updates and eval."""
        return self._encode(rgb_seq, goal, depth_seq)

    # Backwards-compatible thin wrappers.
    def _latent(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_depth:
            return self.base.forward_latent(rgb_seq, goal, depth_seq)
        return self.base.forward_latent(rgb_seq, goal)

    def _mean(self,
              rgb_seq: torch.Tensor,
              goal: torch.Tensor,
              depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_depth:
            return self.base.predict_mean_first(rgb_seq, goal, depth_seq)
        return self.base.predict_mean_first(rgb_seq, goal)

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self,
            rgb_seq: torch.Tensor,
            goal: torch.Tensor,
            depth_seq: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample first-step action; returns (action_z [B,4], log_prob [B], value [B])."""
        latent, mean = self._encode(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return action, log_prob, value

    def act_full(self,
                 rgb_seq: torch.Tensor,
                 goal: torch.Tensor,
                 depth_seq: Optional[torch.Tensor] = None,
                 deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like `act` but also returns the encoder latent. SAC reuses the
        latent to feed Q-networks; saves one backbone forward per call.
        Returns (action_z, log_prob, value, latent).
        """
        latent, mean = self._encode(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return action, log_prob, value, latent

    def evaluate(self,
                 rgb_seq: torch.Tensor,
                 goal: torch.Tensor,
                 action_z: torch.Tensor,
                 depth_seq: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO/SAC update: log_prob, value, entropy for stored actions."""
        latent, mean = self._encode(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        log_prob = dist.log_prob(action_z).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(latent).squeeze(-1)
        return log_prob, value, entropy

    def kl_to(self,
              other: "StochasticVelocityPolicy",
              rgb_seq: torch.Tensor,
              goal: torch.Tensor,
              depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """KL(other || self) per sample, shape [B]. Keeps policy close to reference BC."""
        _, mean = self._encode(rgb_seq, goal, depth_seq)
        with torch.no_grad():
            _, other_mean = other._encode(rgb_seq, goal, depth_seq)
        dist = self._distribution(mean)
        other_dist = other._distribution(other_mean)
        return torch.distributions.kl_divergence(other_dist, dist).sum(dim=-1)

    def frozen_reference_copy(self) -> "StochasticVelocityPolicy":
        """Lean reference for BC/KL anchoring: shares base + log_std with a
        deep-copied snapshot, but skips a critic deepcopy (KL doesn't need it).
        Parameters do not receive gradients.
        """
        # Build empty shell with a deep-copied base + log_std but no extra
        # critic forward path (we override critic to identity-like nn.Module).
        ref = StochasticVelocityPolicy.__new__(StochasticVelocityPolicy)
        nn.Module.__init__(ref)
        ref.base = copy.deepcopy(self.base)
        ref.cmd_dim = self.cmd_dim
        ref.log_std = nn.Parameter(self.log_std.detach().clone(), requires_grad=False)
        # KL never calls critic; use a no-op placeholder.
        ref.critic = nn.Identity()
        ref.use_depth = self.use_depth
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False
        return ref

    def q_input(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Latent features for SAC Q-networks."""
        return self._latent(rgb_seq, goal, depth_seq)


class TwinQCritic(nn.Module):
    """Twin Q-networks for SAC/TD3 on (latent, action_z).

    `layernorm=True` adds LayerNorm after each hidden layer (RLPD, Ball et al.
    2023): it bounds Q-value extrapolation on OOD actions and is the single
    most effective stabilizer for off-policy fine-tuning with high update ratios.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256,
                 layernorm: bool = False) -> None:
        super().__init__()
        in_dim = latent_dim + action_dim

        def _q() -> nn.Sequential:
            layers: list = [nn.Linear(in_dim, hidden)]
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden, hidden))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden, 1))
            return nn.Sequential(*layers)

        self.q1 = _q()
        self.q2 = _q()

    def forward(self, latent: torch.Tensor, action_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent, action_z], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


def load_stochastic_from_checkpoint(
    ckpt_path: Path,
    *,
    init_log_std: float = -0.5,
    device: Optional[torch.device] = None,
    residual_actor: bool = False,
    residual_scale: float = 1.0,
) -> Tuple[StochasticVelocityPolicy, CommandStats, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    base = build_model(cfg)
    base.load_state_dict(ckpt["model"])
    policy = StochasticVelocityPolicy(
        base, init_log_std=init_log_std,
        residual_actor=residual_actor, residual_scale=residual_scale,
    )
    if "rl_head" in ckpt:
        policy.log_std.data.copy_(ckpt["rl_head"]["log_std"])
        policy.critic.load_state_dict(ckpt["rl_head"]["critic"])
        # Resume a residual head if both the ckpt and this policy use one.
        if policy.actor_head is not None and ckpt["rl_head"].get("actor_head") is not None:
            policy.actor_head.load_state_dict(ckpt["rl_head"]["actor_head"])
    stats = CommandStats.from_dict(ckpt["stats"])
    return policy.to(device), stats, cfg


def save_rl_checkpoint(path: Path,
                       policy: StochasticVelocityPolicy,
                       stats: CommandStats,
                       cfg: dict,
                       meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rl_head = {
        "log_std": policy.log_std.detach().cpu(),
        "critic": policy.critic.state_dict(),
    }
    if policy.actor_head is not None:
        rl_head["actor_head"] = policy.actor_head.state_dict()
        rl_head["residual_scale"] = float(policy.residual_scale)
    torch.save(
        {
            "model": policy.base.state_dict(),
            "rl_head": rl_head,
            "stats": stats.to_dict(),
            "config": cfg,
            "rl_meta": meta,
        },
        path,
    )
