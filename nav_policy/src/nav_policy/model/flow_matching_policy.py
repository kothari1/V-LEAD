"""
OT-CFM (Optimal Transport Conditional Flow Matching) policy for goal-conditioned
visuomotor navigation.

Architecture
------------
Encoder (attribute names identical to RGBVelocityPolicy — same state-dict keys):
    ResNet-18 per frame  →  GRU (gru_hidden)  →  LayerNorm  →  h [B, gru_hidden]
    goal_embed: Linear → ReLU → g [B, goal_emb_dim]
    context c = cat([h, g])  [B, feature_dim]

Conditional vector field (the FM-specific head):
    Input:  cat([z_t (H*cmd_dim),  sinusoidal_t (time_emb_dim),  context])
    Hidden: vf_hidden layers, SiLU activation
    Output: H*cmd_dim (velocity field)
    Optional skip: z_t → linear projection added to output

Training — OT-CFM loss (train_fm.py)
--------------------------------------
    x1 = u_star.flatten(1)         ground-truth z-scored horizon [B, H*cmd_dim]
    x0 ~ N(0, I)                   prior sample
    t  ~ U(0, 1)
    x_t = (1-t)*x0 + t*x1         straight OT path
    target_field = x1 - x0
    loss = ||v_θ(x_t, t, encode(rgb, goal)) - target_field||²

Inference — Euler integration
-------------------------------
    x ← N(0, I)
    for i in range(n_steps):
        t = i / n_steps
        x ← x + (1/n_steps) * v_θ(x, t, context)
    reshape x → [B, H, cmd_dim]     (z-scored; de-standardize with CommandStats)

Warm-start compatibility
-------------------------
The encoder keys (visual.*, gru.*, gru_norm.*, goal_embed.*) are identical to
RGBVelocityPolicy, so load_bc_into_feature_extractor() works unchanged — only
vector_field.* keys appear as "unexpected" and are silently ignored.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from nav_policy.model.rgb_velocity_policy import _PerFrameResNet18, count_parameters


# ── Sinusoidal time embedding ─────────────────────────────────────────────────

def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Map continuous time t ∈ [0, 1] to a sinusoidal embedding.

    Args:
        t:   [B] float
        dim: embedding dimension (even recommended)

    Returns: [B, dim] float
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )                                       # [half]
    args = t[:, None] * freqs[None, :]     # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # [B, dim or dim+1]
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ── Conditional vector field MLP ─────────────────────────────────────────────

class ConditionalVectorField(nn.Module):
    """MLP-based conditional vector field v_θ(z_t, t, context) → velocity field.

    Inputs are concatenated: [z_t | t_emb | context] → MLP → action_dim.
    An optional learned skip from z_t adds a residual, which helps the network
    represent near-identity fields at large t (where x_t ≈ x_1).
    """

    def __init__(self,
                 action_dim: int,
                 context_dim: int,
                 time_emb_dim: int = 64,
                 hidden: Sequence[int] = (512, 512, 512),
                 use_skip: bool = True) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.time_emb_dim = time_emb_dim
        self.use_skip = use_skip

        in_dim = action_dim + time_emb_dim + context_dim
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.SiLU())
            last = h
        layers.append(nn.Linear(last, action_dim))
        self.net = nn.Sequential(*layers)

        if use_skip:
            self.skip_proj = nn.Linear(action_dim, action_dim, bias=False)
            nn.init.zeros_(self.skip_proj.weight)

    def forward(self,
                z_t: torch.Tensor,
                t: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t:     [B, action_dim]
            t:       [B]  time ∈ [0, 1]
            context: [B, context_dim]

        Returns: [B, action_dim] predicted velocity field
        """
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim)   # [B, time_emb_dim]
        inp = torch.cat([z_t, t_emb, context], dim=-1)
        v = self.net(inp)
        if self.use_skip:
            v = v + self.skip_proj(z_t)
        return v


# ── Main policy ───────────────────────────────────────────────────────────────

class FlowMatchingPolicy(nn.Module):
    """OT-CFM visuomotor policy.

    The encoder (visual / gru / gru_norm / goal_embed) has the same attribute
    names and parameter shapes as RGBVelocityPolicy so that:
      - load_bc_into_feature_extractor(extractor, fm_ckpt_path) works without
        modification (vector_field.* keys are unexpected, silently ignored).
      - BCEncoderFeatureExtractor's inner_policy.encode() calls FlowMatchingPolicy's
        encode() method via the same interface.
    """

    def __init__(self,
                 T: int = 4,
                 H: int = 10,
                 cmd_dim: int = 4,
                 gru_hidden: int = 256,
                 gru_layers: int = 1,
                 goal_emb_dim: int = 32,
                 goal_input_dim: int = 3,
                 freeze_stem_and_layer1: bool = True,
                 time_emb_dim: int = 64,
                 vf_hidden: Sequence[int] = (512, 512, 512),
                 vf_use_skip: bool = True) -> None:
        super().__init__()
        if goal_input_dim not in (2, 3):
            raise ValueError(f"goal_input_dim must be 2 or 3; got {goal_input_dim}")

        self.T = T
        self.H = H
        self.cmd_dim = cmd_dim
        self.gru_hidden = gru_hidden
        self.goal_emb_dim = goal_emb_dim
        self.goal_input_dim = int(goal_input_dim)

        # ── Encoder — attribute names match RGBVelocityPolicy exactly ──────
        self.visual = _PerFrameResNet18(freeze_stem_and_layer1=freeze_stem_and_layer1)
        self.gru = nn.GRU(
            input_size=self.visual.out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.gru_norm = nn.LayerNorm(gru_hidden)
        self.goal_embed = nn.Sequential(
            nn.Linear(self.goal_input_dim, goal_emb_dim),
            nn.ReLU(inplace=True),
        )

        # ── Conditional vector field (FM head) ──────────────────────────────
        self.vector_field = ConditionalVectorField(
            action_dim=self.action_dim,
            context_dim=self.feature_dim,
            time_emb_dim=time_emb_dim,
            hidden=tuple(vf_hidden),
            use_skip=vf_use_skip,
        )

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Encoder output size; same as RGBVelocityPolicy.feature_dim."""
        return self.gru_hidden + self.goal_emb_dim

    @property
    def action_dim(self) -> int:
        return self.H * self.cmd_dim

    # ── Encoder (identical signature to RGBVelocityPolicy.encode) ────────────

    def encode(self,
               rgb_seq: torch.Tensor,
               goal: torch.Tensor) -> torch.Tensor:
        """Visual + goal encoder.

        Signature identical to RGBVelocityPolicy.encode() so BCEncoderFeatureExtractor
        can call this transparently via inner_policy.encode().

        Args:
            rgb_seq: [B, T, 3, S, S] float32, ImageNet-normalized
            goal:    [B, goal_input_dim] float32

        Returns: [B, feature_dim]
        """
        if rgb_seq.ndim != 5:
            raise ValueError(f"expected rgb_seq [B,T,3,S,S], got {tuple(rgb_seq.shape)}")
        B, T, C, S1, S2 = rgb_seq.shape
        if T != self.T:
            raise ValueError(f"T mismatch: config={self.T}, input={T}")
        if goal.shape != (B, self.goal_input_dim):
            raise ValueError(f"goal must be [B,{self.goal_input_dim}], got {tuple(goal.shape)}")

        flat = rgb_seq.reshape(B * T, C, S1, S2)
        feats = self.visual(flat)
        seq = feats.view(B, T, self.visual.out_dim)

        _, h_n = self.gru(seq)
        h = self.gru_norm(h_n[-1])

        g = self.goal_embed(goal)
        return torch.cat([h, g], dim=-1)

    # ── Training ──────────────────────────────────────────────────────────────

    def fm_loss(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                u_star: torch.Tensor) -> torch.Tensor:
        """OT-CFM matching loss (scalar).

        Args:
            rgb_seq: [B, T, 3, S, S] ImageNet-normalized float32
            goal:    [B, goal_input_dim] float32
            u_star:  [B, H, cmd_dim] float32, z-scored ground truth

        Returns: scalar loss tensor
        """
        B = u_star.shape[0]
        x1 = u_star.reshape(B, self.action_dim)

        x0 = torch.randn(B, self.action_dim, device=x1.device, dtype=x1.dtype)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)

        t_bc = t[:, None]
        x_t = (1.0 - t_bc) * x0 + t_bc * x1
        target = x1 - x0

        context = self.encode(rgb_seq, goal)
        v_pred = self.vector_field(x_t, t, context)
        return F.mse_loss(v_pred, target)

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self,
               rgb_seq: torch.Tensor,
               goal: torch.Tensor,
               n_steps: int = 10,
               generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Euler ODE integration from x0 ~ N(0,I) to predicted action horizon.

        Args:
            rgb_seq:   [B, T, 3, S, S] ImageNet-normalized float32
            goal:      [B, goal_input_dim] float32
            n_steps:   Euler steps; higher → better accuracy. OT-CFM learns
                       near-straight paths so n_steps=4 is often sufficient.
            generator: optional RNG for reproducible sampling

        Returns:
            commands: [B, H, cmd_dim] float32, z-scored
                      (de-standardize with CommandStats to get physical units)
        """
        B = rgb_seq.shape[0]
        context = self.encode(rgb_seq, goal)

        x = torch.randn(
            B, self.action_dim,
            device=rgb_seq.device,
            dtype=rgb_seq.dtype,
            generator=generator,
        )

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=x.device, dtype=x.dtype)
            v = self.vector_field(x, t, context)
            x = x + dt * v

        return x.reshape(B, self.H, self.cmd_dim)
