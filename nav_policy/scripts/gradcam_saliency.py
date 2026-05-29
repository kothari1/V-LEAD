#!/usr/bin/env python3
"""
Generate Grad-CAM saliency overlays for ResNet layer4 in RGB / RGB+DA2 policies.

Usage (inside Docker, from nav_policy/):
    python scripts/gradcam_saliency.py \\
        --checkpoint data/checkpoints_flightroom/bc_best.pt \\
        --cache data/processed_flightroom/cache/flightroom_ssv_exp_2026-05-22_071733_trajs-110/file00000_sub0.pt \\
        --k 50 \\
        --output-dir data/eval/saliency_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nav_policy.data.normalization import CommandStats, imagenet_normalize
from nav_policy.model.factory import build_model, model_uses_depth


def _load_goal(blob: dict, k: int, goal_input_dim: int, scale: float) -> torch.Tensor:
    heading = blob["goal_heading"][k].float()
    if goal_input_dim == 2:
        return heading
    d = float(blob["goal_dist"][k].item()) / scale
    return torch.cat([heading, torch.tensor([d], dtype=torch.float32)])


def main() -> None:
    p = argparse.ArgumentParser(description="Grad-CAM saliency for nav_policy checkpoints.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--k", type=int, default=50, help="Window end index in cache")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame-index", type=int, default=-1, help="Which of T frames (-1=last)")
    args = p.parse_args()

    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        raise SystemExit(
            "pytorch-grad-cam is required: pip install pytorch-grad-cam"
        ) from exc

    ckpt = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = ckpt["config"]
    T = int(cfg["window"]["T"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).eval().to(device)
    model.load_state_dict(ckpt["model"])
    stats = CommandStats.from_dict(ckpt["stats"])
    uses_depth = model_uses_depth(cfg)
    goal_input_dim = int(cfg["model"].get("goal_input_dim", 3))
    goal_scale = float(cfg["model"].get("goal_distance_scale", 5.0))

    blob = torch.load(args.cache, weights_only=False, map_location="cpu")
    k = int(args.k)
    rgb_u8 = blob["rgb"][k - T + 1 : k + 1]
    fi = args.frame_index if args.frame_index >= 0 else T - 1
    frame_u8 = rgb_u8[fi].float() / 255.0
    rgb_norm = imagenet_normalize(rgb_u8, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    rgb_seq = rgb_norm.unsqueeze(0).to(device)
    goal = _load_goal(blob, k, goal_input_dim, goal_scale).unsqueeze(0).to(device)

    depth_seq = None
    if uses_depth:
        if "depth" not in blob:
            raise RuntimeError("cache missing depth; run precompute_da2_depth.py")
        dep = blob["depth"][k - T + 1 : k + 1].float() / 255.0
        depth_seq = dep.unsqueeze(0).to(device)

    target_layer = model.visual.backbone.layer4

    class _Wrap(torch.nn.Module):
        def __init__(self, m, g, d):
            super().__init__()
            self.m = m
            self.g = g
            self.d = d

        def forward(self, x):
            if self.d is not None:
                return self.m(x, self.g, self.d)
            return self.m(x, self.g)

    wrap = _Wrap(model, goal, depth_seq)
    cam = GradCAM(model=wrap, target_layers=[target_layer])

    input_tensor = rgb_seq[:, fi]
    grayscale = cam(input_tensor=input_tensor)[0]
    rgb_hwc = frame_u8.permute(1, 2, 0).numpy()
    overlay = show_cam_on_image(rgb_hwc, grayscale, use_rgb=True)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v3 as iio
        iio.imwrite(str(out_dir / "gradcam_overlay.png"), overlay.astype(np.uint8))
        iio.imwrite(str(out_dir / "rgb_frame.png"), (rgb_hwc * 255).astype(np.uint8))
    except Exception:
        from PIL import Image
        Image.fromarray(overlay.astype(np.uint8)).save(out_dir / "gradcam_overlay.png")
        Image.fromarray((rgb_hwc * 255).astype(np.uint8)).save(out_dir / "rgb_frame.png")

    meta = {
        "checkpoint": str(args.checkpoint),
        "cache": str(args.cache),
        "k": k,
        "frame_index": fi,
        "arch": cfg.get("model", {}).get("arch", "rgb_resnet18"),
    }
    (out_dir / "meta.yaml").write_text(yaml.safe_dump(meta))
    print(f"[gradcam] wrote overlays -> {out_dir}")


if __name__ == "__main__":
    main()
