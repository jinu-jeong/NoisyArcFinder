"""
predict_heatmap.py
──────────────────
Load checkpoints/heatmap_best.pt and predict centroids for all train/test images.
Saves visualizations to prediction_result/{train,test}/.

Visualization per image:
  - Red circle          : GT bulb circle (radius from ground_truth.json)
  - Red sniper crosshair: GT centroid  (lines from circle edge toward center w/ gap)
  - Green X             : predicted centroid
  - Top-left text       : mean pixel error for this image
"""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_heatmap import (
    CKPT, DEVICE, IN_H, IN_W, ORIG_H, ORIG_W,
    UNet, decode_heatmap, sort_by_x, to_orig_coords,
)

OUT_DIR = "prediction_result"
SPLITS  = [
    ("train", "synthetic_data/train",  os.path.join(OUT_DIR, "train")),
    ("test",  "synthetic_data/test",   os.path.join(OUT_DIR, "test")),
]


# ── Drawing helpers ────────────────────────────────────────
def draw_sniper_crosshair(img, cx, cy, radius, color=(0, 0, 255), thickness=2):
    """GT marker: circle + sniper-scope crosshair (lines from circle edge inward)."""
    icx, icy, r = int(round(cx)), int(round(cy)), max(1, int(round(radius)))
    cv2.circle(img, (icx, icy), r, color, thickness)
    gap = max(4, r // 4)
    lw  = max(1, thickness - 1)
    # Left arm
    cv2.line(img, (icx - r, icy), (icx - gap, icy), color, lw)
    # Right arm
    cv2.line(img, (icx + gap, icy), (icx + r, icy), color, lw)
    # Top arm
    cv2.line(img, (icx, icy - r), (icx, icy - gap), color, lw)
    # Bottom arm
    cv2.line(img, (icx, icy + gap), (icx, icy + r), color, lw)


def draw_x(img, cx, cy, size=9, color=(0, 255, 0), thickness=2):
    """Predicted centroid: X mark."""
    icx, icy = int(round(cx)), int(round(cy))
    cv2.line(img, (icx - size, icy - size), (icx + size, icy + size), color, thickness)
    cv2.line(img, (icx + size, icy - size), (icx - size, icy + size), color, thickness)


# ── Inference + visualization ──────────────────────────────
def predict_split(model, split_name, data_dir, out_dir, transform):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(data_dir, "ground_truth.json")) as f:
        records = json.load(f)

    total_err  = 0.0
    all_errors = []

    for rec in records:
        img_path = os.path.join(data_dir, rec["image"])

        # ── Inference ─────────────────────────────────────
        pil_img = Image.open(img_path).convert("RGB")
        inp     = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            hm = model(inp)

        pts_out  = decode_heatmap(hm)
        pts_orig = sort_by_x(to_orig_coords(pts_out))[0]  # (3, 2)

        # ── Pixel error for this image ─────────────────────
        sorted_gt_pts = sorted(
            [(c["x"], c["y"]) for c in rec["centroids"]], key=lambda p: p[0]
        )
        gt_tensor  = torch.tensor(sorted_gt_pts, dtype=torch.float32)  # (3, 2)
        img_err    = (pts_orig - gt_tensor).norm(dim=-1).mean().item()
        total_err += img_err
        all_errors.append(img_err)

        # ── Draw ──────────────────────────────────────────
        vis = cv2.imread(img_path)  # BGR, ORIG_W × ORIG_H

        # GT: red circle + sniper crosshair
        sorted_gt = sorted(rec["centroids"], key=lambda c: c["x"])
        for c in sorted_gt:
            draw_sniper_crosshair(vis, c["x"], c["y"],
                                  radius=c.get("radius", 30),
                                  color=(0, 0, 255))

        # Predicted: green X
        for px, py in pts_orig.tolist():
            draw_x(vis, px, py, color=(0, 255, 0))

        # Per-image pixel error text
        cv2.putText(vis, f"err={img_err:.1f}px", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, rec["image"]), vis)

    mean_err = total_err / len(records)
    print(f"[{split_name}]  {len(records)} images  |  "
          f"mean px err = {mean_err:.2f}px  |  "
          f"max = {max(all_errors):.1f}px  |  "
          f"saved → {out_dir}/")


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    # Build transform (same as training, no augmentation)
    transform = transforms.Compose([
        transforms.Resize((IN_H, IN_W)),
        transforms.ToTensor(),
    ])

    # Load model
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CKPT}  (device={DEVICE})\n")

    for split_name, data_dir, out_dir in SPLITS:
        predict_split(model, split_name, data_dir, out_dir, transform)
