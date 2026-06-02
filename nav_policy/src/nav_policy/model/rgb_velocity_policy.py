"""
ResNet-18 + GRU + MLP policy that maps an RGB frame sequence and a goal vector
to a horizon of velocity commands [vx, vy, vz, psi_dot].

Forward shapes:
    rgb_seq: [B, T, 3, S, S]  (S = 224 by default, ImageNet-normalized)
    goal:    [B, goal_input_dim]
                goal_input_dim = 2 -> [hx, hy]               (heading only)
                goal_input_dim = 3 -> [hx, hy, d_normalized] (heading + distance)
    output:  [B, H, cmd_dim]  (z-scored; caller de-standardizes with CommandStats)

Architectural notes:
    - ResNet-18 (shared weights across time) encodes each frame to a 512-D vector.
    - A GRU processes the T-frame visual sequence and yields a context vector h.
    - LayerNorm is applied to h before concatenation, normalizing its magnitude so
      the goal embedding (which is naturally unit-scale from a unit-vector input)
      contributes meaningfully and neither stream drowns out the other.
    - The goal vector is projected through a small linear embedding (goal_emb_dim).
    - [h_normed ; goal_emb] are concatenated and fed to the MLP head.
    - Dropout after each hidden ReLU in the MLP head regularizes the prediction head
      without touching the backbone or GRU.
    - The stem (`conv1` + `bn1`) and `layer1` can optionally be frozen for early
      training stability on small datasets / small GPUs.
    - The 3-dim goal-input variant adds a scale-normalized distance-to-goal scalar
      alongside the unit heading vector, giving the network a direct signal for
      deceleration as it approaches the goal.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class _PerFrameResNet18(nn.Module):
    """ResNet-18 with the final classifier removed; returns a 512-D pooled feature per image."""

    def __init__(self, freeze_stem_and_layer1: bool = True) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Replace the FC head with identity -> .forward() returns the 512-D pooled feature.
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.out_dim = 512

        if freeze_stem_and_layer1:
            for m in (self.backbone.conv1, self.backbone.bn1, self.backbone.layer1):
                for p in m.parameters():
                    p.requires_grad = False
            # Keep BN in eval mode so its running stats don't drift on small batches.
            self.backbone.bn1.eval()
            for bn_module in self.backbone.layer1.modules():
                if isinstance(bn_module, nn.BatchNorm2d):
                    bn_module.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class _MLPHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden: Sequence[int],
                 out_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthEncoder(nn.Module):
    """Small CNN encoding a single-channel depth map to a 256-D vector.
    Matches BC checkpoint depth_enc architecture exactly."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),   # net.0
            nn.ReLU(inplace=True),              # net.1
            nn.Conv2d(32, 64, 3, padding=1),   # net.2
            nn.ReLU(inplace=True),              # net.3
            nn.Conv2d(64, 128, 3, padding=1),  # net.4
            nn.ReLU(inplace=True),              # net.5
            nn.AdaptiveAvgPool2d(1),            # net.6
            nn.Flatten(),                        # net.7
            nn.Linear(128, 256),                # net.8
        )
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RGBDepthFusion(nn.Module):
    """Cross-attention fusion: RGB (512D) queries depth (256D→512D).
    Matches BC checkpoint fusion architecture exactly."""

    def __init__(self, rgb_dim: int = 512, dep_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.rgb_norm = nn.LayerNorm(rgb_dim)
        self.dep_norm = nn.LayerNorm(dep_dim)
        self.dep_to_rgb = nn.Linear(dep_dim, rgb_dim)
        self.attn = nn.MultiheadAttention(rgb_dim, num_heads=num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(rgb_dim)
        self.out_dim = rgb_dim

    def forward(self, rgb_feat: torch.Tensor, dep_feat: torch.Tensor) -> torch.Tensor:
        rgb_n = self.rgb_norm(rgb_feat)
        dep_proj = self.dep_to_rgb(self.dep_norm(dep_feat))
        q = rgb_n.unsqueeze(1)
        kv = dep_proj.unsqueeze(1)
        attn_out, _ = self.attn(q, kv, kv)
        return self.out_norm(attn_out.squeeze(1))


class RGBVelocityPolicy(nn.Module):
    """
    RGB sequence + goal vector -> velocity command horizon.

    Goal vector is one of:
        * [hx, hy]                     -- unit heading toward the goal (goal_input_dim=2)
        * [hx, hy, d_normalized]       -- unit heading + scale-normalized distance (3)

    It is projected through a small linear embedding and concatenated with
    the LayerNorm'd GRU context before the MLP prediction head, giving the
    policy an explicit, direction-aware navigation objective without changing
    the visual backbone or recurrent structure.

    Outputs z-scored values; de-standardize with CommandStats.
    """

    def __init__(self,
                 T: int = 4,
                 H: int = 10,
                 cmd_dim: int = 4,
                 gru_hidden: int = 256,
                 gru_layers: int = 1,
                 mlp_hidden: Sequence[int] = (256, 128),
                 mlp_dropout: float = 0.1,
                 goal_emb_dim: int = 32,
                 goal_input_dim: int = 2,
                 freeze_stem_and_layer1: bool = True,
                 use_depth: bool = False) -> None:
        super().__init__()
        if goal_input_dim not in (2, 3):
            raise ValueError(f"goal_input_dim must be 2 or 3; got {goal_input_dim}")
        self.T = T
        self.H = H
        self.cmd_dim = cmd_dim
        self.gru_hidden = gru_hidden
        self.goal_emb_dim = goal_emb_dim
        self.goal_input_dim = int(goal_input_dim)
        self.use_depth = use_depth

        self.visual = _PerFrameResNet18(freeze_stem_and_layer1=freeze_stem_and_layer1)

        # Depth encoder + fusion (optional, matches BC checkpoint architecture).
        if use_depth:
            self.depth_enc = DepthEncoder()
            self.fusion = RGBDepthFusion(rgb_dim=self.visual.out_dim, dep_dim=self.depth_enc.out_dim)
            gru_input_size = self.fusion.out_dim
        else:
            gru_input_size = self.visual.out_dim

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        # LayerNorm on the GRU output normalizes its magnitude so the goal
        # embedding (naturally ~unit-scale from a unit-vector input) contributes
        # proportionally when the two streams are concatenated.
        self.gru_norm = nn.LayerNorm(gru_hidden)

        # Goal embedding: goal_input_dim -> goal_emb_dim.
        self.goal_embed = nn.Sequential(
            nn.Linear(self.goal_input_dim, goal_emb_dim),
            nn.ReLU(inplace=True),
        )
        # MLP head: concatenation of normalized GRU context + goal embedding.
        self.head = _MLPHead(
            in_dim=gru_hidden + goal_emb_dim,
            hidden=tuple(mlp_hidden),
            out_dim=H * cmd_dim,
            dropout=mlp_dropout,
        )

    @property
    def feature_dim(self) -> int:
        return self.gru_hidden + self.goal_emb_dim

    def encode(self,
               rgb_seq: torch.Tensor,
               goal: torch.Tensor,
               depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Shared visual+goal encoder. Returns fused feature [B, feature_dim].

        Args:
            rgb_seq:   [B, T, 3, S, S] float32, ImageNet-normalized.
            goal:      [B, goal_input_dim] float32.
            depth_seq: [B, T, 1, S, S] float32 metric depth (optional, requires use_depth=True).
        """
        if rgb_seq.ndim != 5:
            raise ValueError(f"expected rgb_seq [B,T,3,S,S], got {tuple(rgb_seq.shape)}")
        B, T, C, S1, S2 = rgb_seq.shape
        if T != self.T:
            raise ValueError(f"T mismatch: config={self.T}, input={T}")
        if goal.shape != (B, self.goal_input_dim):
            raise ValueError(
                f"goal must be [B,{self.goal_input_dim}], got {tuple(goal.shape)}"
            )

        flat = rgb_seq.reshape(B * T, C, S1, S2)
        feats = self.visual(flat)  # [B*T, 512]

        if self.use_depth and depth_seq is not None:
            dep_flat = depth_seq.reshape(B * T, *depth_seq.shape[2:])  # [B*T, 1, S, S]
            dep_feats = self.depth_enc(dep_flat)                        # [B*T, 256]
            feats = self.fusion(feats, dep_feats)                       # [B*T, 512]

        seq = feats.view(B, T, -1)
        _, h_n = self.gru(seq)
        h = self.gru_norm(h_n[-1])

        g = self.goal_embed(goal)
        return torch.cat([h, g], dim=-1)

    def forward(self,
                rgb_seq: torch.Tensor,
                goal: torch.Tensor,
                depth_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = rgb_seq.shape[0]
        h_aug = self.encode(rgb_seq, goal, depth_seq=depth_seq)
        out = self.head(h_aug)
        return out.view(B, self.H, self.cmd_dim)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
