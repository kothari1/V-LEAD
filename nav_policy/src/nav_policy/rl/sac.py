"""Off-policy actor-critic updates for nav_policy RL fine-tuning.

Two modes share one trainer (twin-Q critic + replay):

* ``td3bc=False`` — original SAC (entropy-regularized stochastic actor). Kept
  for back-compat; this is the path that catastrophically forgot the BC prior
  on the long run (cold random critic drives a warm BC actor off-distribution).

* ``td3bc=True``  — TD3+BC stable fine-tuner (Fujimoto & Gu, 2021) for the
  residual-actor policy. Deterministic actor, twin-Q min, delayed + smoothed
  target-actor updates, an action-space BC anchor (``MSE(pi, bc_mean)``, i.e.
  penalize the residual), a critic-only warmup phase so Q is informed before it
  moves the actor, and LayerNorm critics (RLPD). Designed so the policy starts
  exactly at BC and only deviates where Q justifies it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from nav_policy.rl.stochastic_policy import StochasticVelocityPolicy, TwinQCritic


@dataclass
class SACStats:
    q1_loss: float
    q2_loss: float
    policy_loss: float
    alpha: float
    ref_kl: float = 0.0
    bc_mse: float = 0.0


class SACTrainer:
    """SAC / TD3+BC trainer operating in z-score action space."""

    def __init__(self,
                 policy: StochasticVelocityPolicy,
                 *,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 auto_alpha: bool = True,
                 ref_kl_coef: float = 0.0,
                 reference_policy: Optional[StochasticVelocityPolicy] = None,
                 device: torch.device,
                 # --- TD3+BC knobs (only used when td3bc=True) ---
                 td3bc: bool = False,
                 td3bc_alpha: float = 2.5,
                 bc_weight: float = 1.0,
                 td3bc_normalize: bool = True,
                 policy_delay: int = 2,
                 target_policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 action_clip: float = 4.0,
                 critic_warmup_updates: int = 0,
                 critic_layernorm: bool = False) -> None:
        self.policy = policy
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.ref_kl_coef = float(ref_kl_coef)
        self.reference_policy = reference_policy

        self.td3bc = bool(td3bc)
        self.td3bc_alpha = float(td3bc_alpha)
        self.bc_weight = float(bc_weight)
        self.td3bc_normalize = bool(td3bc_normalize)
        self.policy_delay = max(1, int(policy_delay))
        self.target_policy_noise = float(target_policy_noise)
        self.noise_clip = float(noise_clip)
        self.action_clip = float(action_clip)
        self.critic_warmup_updates = int(critic_warmup_updates)
        self._update_count = 0

        if self.td3bc and policy.actor_head is None:
            raise ValueError(
                "td3bc=True requires a residual-actor policy "
                "(residual_actor=True); got policy.actor_head=None."
            )

        latent_dim = int(policy.base.gru_hidden) + int(policy.base.goal_emb_dim)
        self.critic = TwinQCritic(
            latent_dim, policy.cmd_dim, layernorm=critic_layernorm,
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Target actor (TD3): only the residual head moves, so a lean polyak
        # copy of the head suffices (the BC base + bc_mean are frozen/shared).
        if self.td3bc:
            self.target_actor_head = copy.deepcopy(policy.actor_head)
            for p in self.target_actor_head.parameters():
                p.requires_grad = False
        else:
            self.target_actor_head = None

        # Only optimize trainable params (frozen base must not enter the opt).
        actor_params = [p for p in policy.parameters() if p.requires_grad]
        self.policy_opt = torch.optim.Adam(actor_params, lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if auto_alpha and not self.td3bc:
            # log_alpha is a scalar (shape ()) — broadcasts cleanly against
            # per-sample log_prob without an extra squeeze.
            self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
            self.target_entropy = -float(policy.cmd_dim)
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.target_entropy = 0.0
            self._fixed_alpha = alpha

    @property
    def alpha(self) -> torch.Tensor:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        return torch.tensor(self._fixed_alpha, device=self.device)

    def _soft_update(self) -> None:
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        if self.target_actor_head is not None:
            for p, pt in zip(self.policy.actor_head.parameters(),
                             self.target_actor_head.parameters()):
                pt.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ---- batch unpack -------------------------------------------------------
    def _unpack(self, batch: dict):
        rgb = batch["rgb"].to(self.device).float()
        goal = batch["goal"].to(self.device).float()
        depth = (
            batch["depth"].to(self.device).float()
            if batch.get("depth") is not None else None
        )
        next_rgb = batch["next_rgb"].to(self.device).float()
        next_goal = batch["next_goal"].to(self.device).float()
        next_depth = (
            batch["next_depth"].to(self.device).float()
            if batch.get("next_depth") is not None else None
        )
        actions = batch["actions"].to(self.device).float()
        rewards = batch["rewards"].to(self.device).float()
        dones = batch["dones"].to(self.device).float()
        return rgb, goal, depth, next_rgb, next_goal, next_depth, actions, rewards, dones

    def update(self, batch: dict) -> SACStats:
        # cuDNN refuses RNN backward when the GRU is in eval() mode; mirror
        # PPO's pattern and put the policy in train() before the backward.
        self.policy.train()
        if self.td3bc:
            return self._update_td3bc(batch)
        return self._update_sac(batch)

    # ---- TD3+BC -------------------------------------------------------------
    def _td3bc_target_action(self, next_rgb, next_goal, next_depth):
        """Smoothed target action via the target residual head (no grad)."""
        latent_next = self.policy.q_input(next_rgb, next_goal, next_depth)
        bc_next = self.policy.bc_mean(next_rgb, next_goal, next_depth)
        resid = self.policy.residual_scale * torch.tanh(
            self.target_actor_head(latent_next)
        )
        mean_next = bc_next + resid
        noise = (torch.randn_like(mean_next) * self.target_policy_noise).clamp(
            -self.noise_clip, self.noise_clip
        )
        a_next = (mean_next + noise).clamp(-self.action_clip, self.action_clip)
        return latent_next, a_next

    def _update_td3bc(self, batch: dict) -> SACStats:
        (rgb, goal, depth, next_rgb, next_goal, next_depth,
         actions, rewards, dones) = self._unpack(batch)

        # --- critic update (every step) ---
        with torch.no_grad():
            latent_next, a_next = self._td3bc_target_action(
                next_rgb, next_goal, next_depth,
            )
            q1_t, q2_t = self.critic_target(latent_next, a_next)
            q_next = torch.min(q1_t, q2_t)
            target = rewards + (1.0 - dones) * self.gamma * q_next
            # Frozen base -> the encoder latent is a constant input to the
            # critic; no grad needed (critic optimizes only its own params).
            latent = self.policy.q_input(rgb, goal, depth)

        q1, q2 = self.critic(latent, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        self._update_count += 1
        policy_loss_val = 0.0
        bc_mse_val = 0.0

        # --- delayed actor update, gated by critic warmup ---
        warmed = self._update_count > self.critic_warmup_updates
        if warmed and (self._update_count % self.policy_delay == 0):
            latent_pi, mean = self.policy.deterministic_mean(rgb, goal, depth)
            # Detach the encoder latent feeding the critic so the actor gradient
            # flows only through the action, not back into the (frozen) encoder.
            q1_pi, _ = self.critic(latent_pi.detach(), mean)
            with torch.no_grad():
                bc_target = self.policy.bc_mean(rgb, goal, depth)
            bc_mse = F.mse_loss(mean, bc_target)

            if self.td3bc_normalize:
                lmbda = self.td3bc_alpha / (q1_pi.abs().mean().detach() + 1e-6)
            else:
                lmbda = self.td3bc_alpha
            policy_loss = -lmbda * q1_pi.mean() + self.bc_weight * bc_mse

            self.policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_opt.step()

            policy_loss_val = float(policy_loss.item())
            bc_mse_val = float(bc_mse.item())

        self._soft_update()
        return SACStats(
            q1_loss=float(q1_loss.item()),
            q2_loss=float(q2_loss.item()),
            policy_loss=policy_loss_val,
            alpha=0.0,
            ref_kl=0.0,
            bc_mse=bc_mse_val,
        )

    # ---- original SAC (back-compat) ----------------------------------------
    def _update_sac(self, batch: dict) -> SACStats:
        (rgb, goal, depth, next_rgb, next_goal, next_depth,
         actions, rewards, dones) = self._unpack(batch)

        with torch.no_grad():
            # act_full returns the encoder latent so we don't pay another
            # backbone forward for q_input.
            next_actions, next_log_prob, _, latent_next = self.policy.act_full(
                next_rgb, next_goal, next_depth,
            )
            q1_next, q2_next = self.critic_target(latent_next, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target = rewards + (1.0 - dones) * self.gamma * q_next

        latent = self.policy.q_input(rgb, goal, depth)
        q1, q2 = self.critic(latent, actions)
        q1_loss = F.mse_loss(q1, target)
        q2_loss = F.mse_loss(q2, target)
        critic_loss = q1_loss + q2_loss

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_prob, _, latent = self.policy.act_full(rgb, goal, depth)
        q1_pi, q2_pi = self.critic(latent, new_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self.alpha.detach()
        policy_loss = (alpha * log_prob - q_pi).mean()

        ref_kl_val = 0.0
        if self.reference_policy is not None and self.ref_kl_coef > 0.0:
            ref_kl = self.policy.kl_to(self.reference_policy, rgb, goal, depth).mean()
            policy_loss = policy_loss + self.ref_kl_coef * ref_kl
            ref_kl_val = float(ref_kl.item())

        self.policy_opt.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_opt.step()

        alpha_val = float(alpha.item())
        if self.log_alpha is not None and self.alpha_opt is not None:
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            alpha_val = float(self.alpha.item())

        self._soft_update()
        return SACStats(
            q1_loss=float(q1_loss.item()),
            q2_loss=float(q2_loss.item()),
            policy_loss=float(policy_loss.item()),
            alpha=alpha_val,
            ref_kl=ref_kl_val,
        )


def sac_config_from_dict(cfg: Dict) -> Dict:
    s = cfg.get("sac", {}) or {}
    anchor = cfg.get("bc_anchor", {}) or {}
    ref_kl_coef = s.get("ref_kl_coef", anchor.get("kl_coef", 0.0))
    return {
        "lr": float(s.get("lr", 3e-4)),
        "gamma": float(s.get("gamma", 0.99)),
        "tau": float(s.get("tau", 0.005)),
        "alpha": float(s.get("alpha", 0.2)),
        "auto_alpha": bool(s.get("auto_alpha", True)),
        "ref_kl_coef": float(ref_kl_coef),
        "batch_size": int(s.get("batch_size", 256)),
        "updates_per_iter": int(s.get("updates_per_iter", 4)),
        "replay_capacity": int(s.get("replay_capacity", 100_000)),
        # TD3+BC
        "td3bc": bool(s.get("td3bc", False)),
        "td3bc_alpha": float(s.get("td3bc_alpha", 2.5)),
        "bc_weight": float(s.get("bc_weight", 1.0)),
        "td3bc_normalize": bool(s.get("td3bc_normalize", True)),
        "policy_delay": int(s.get("policy_delay", 2)),
        "target_policy_noise": float(s.get("target_policy_noise", 0.2)),
        "noise_clip": float(s.get("noise_clip", 0.5)),
        "action_clip": float(s.get("action_clip", 4.0)),
        "critic_warmup_updates": int(s.get("critic_warmup_updates", 0)),
        "critic_layernorm": bool(s.get("critic_layernorm", False)),
        "residual_actor": bool(s.get("residual_actor", False)),
        "residual_scale": float(s.get("residual_scale", 1.0)),
    }
