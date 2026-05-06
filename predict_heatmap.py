"""
predict_heatmap.py
──────────────────
Load checkpoints/heatmap_best.pt and predict centroids for all train/test images.
Saves visualizations to prediction_result/{train,test}/.

Visualization per image:
  - Red circle          : GT bulb circle (radius from ground_truth.json)
  - Red sniper crosshair: GT centroid  (lines from circle edge toward center w/ gap)
  - Green X             : predicted centroid
  - Top-left text       : absolute error (px) and relative error (% of image diagonal)
                          All sizes (font, markers) scale proportionally with image width.
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
    ("val",   "synthetic_data/val",    os.path.join(OUT_DIR, "val")),
    ("test",  "synthetic_data/test",   os.path.join(OUT_DIR, "test")),
]

# Reference width for scaling all drawn sizes (markers, font, thickness)
REF_W = 640.0


# ── Drawing helpers ────────────────────────────────────────
def draw_sniper_crosshair(img, cx, cy, radius, color=(0, 0, 255), thickness=2):
    """GT marker: circle + full crosshair through center (edge to edge, no gap)."""
    icx, icy, r = int(round(cx)), int(round(cy)), max(1, int(round(radius)))
    cv2.circle(img, (icx, icy), r, color, thickness)
    lw = max(1, thickness - 1)
    # Horizontal: left edge → right edge
    cv2.line(img, (icx - r, icy), (icx + r, icy), color, lw)
    # Vertical: top edge → bottom edge
    cv2.line(img, (icx, icy - r), (icx, icy + r), color, lw)


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

    total_abs_err = 0.0
    total_rel_err = 0.0
    all_abs_errors = []
    all_rel_errors = []

    for rec in records:
        img_path = os.path.join(data_dir, rec["image"])

        # ── Inference ─────────────────────────────────────
        pil_img = Image.open(img_path).convert("RGB")
        inp     = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            hm = model(inp)

        orig_w, orig_h = pil_img.size   # actual image dimensions (may vary)
        orig_wh = torch.tensor([[orig_w, orig_h]], dtype=torch.float32)
        pts_out  = decode_heatmap(hm)
        pts_orig = sort_by_x(to_orig_coords(pts_out, orig_wh))[0]  # (3, 2)

        # ── Errors: absolute (px) and relative (% of image diagonal) ─────
        sorted_gt_pts = sorted(
            [(c["x"], c["y"]) for c in rec["centroids"]], key=lambda p: p[0]
        )
        gt_tensor = torch.tensor(sorted_gt_pts, dtype=torch.float32)  # (3, 2)
        img_abs   = (pts_orig - gt_tensor).norm(dim=-1).mean().item()
        diag      = (orig_w ** 2 + orig_h ** 2) ** 0.5
        img_rel   = img_abs / diag * 100   # % of diagonal

        total_abs_err += img_abs
        total_rel_err += img_rel
        all_abs_errors.append(img_abs)
        all_rel_errors.append(img_rel)

        # ── Draw ──────────────────────────────────────────
        vis   = cv2.imread(img_path)
        scale = orig_w / REF_W   # linear scale relative to 640 px reference

        marker_thick = max(1, int(round(2 * scale)))
        x_size       = max(4, int(round(9 * scale)))

        # GT: red circle + sniper crosshair
        sorted_gt = sorted(rec["centroids"], key=lambda c: c["x"])
        for c in sorted_gt:
            draw_sniper_crosshair(vis, c["x"], c["y"],
                                  radius=c.get("radius", 30),
                                  color=(0, 0, 255),
                                  thickness=marker_thick)

        # Predicted: green X
        for px, py in pts_orig.tolist():
            draw_x(vis, px, py, size=x_size, color=(0, 255, 0),
                   thickness=marker_thick)

        # Per-image error text — two lines, font scaled with image width
        font_scale = 0.80 * scale          # 0.55 was too small at REF_W; use 0.80
        font_thick = max(1, int(round(1.5 * scale)))
        margin     = max(6, int(round(8  * scale)))
        line_h     = max(16, int(round(30 * scale)))

        cv2.putText(vis,
                    f"abs err = {img_abs:.1f} px",
                    (margin, line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 220, 255), font_thick, cv2.LINE_AA)
        cv2.putText(vis,
                    f"rel err = {img_rel:.3f}% diag",
                    (margin, line_h * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 220, 255), font_thick, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, rec["image"]), vis)

    n = len(records)
    mean_abs = total_abs_err / n
    mean_rel = total_rel_err / n
    print(f"[{split_name}]  {n} images  |  "
          f"mean abs = {mean_abs:.2f}px  max abs = {max(all_abs_errors):.1f}px  |  "
          f"mean rel = {mean_rel:.3f}%diag  max rel = {max(all_rel_errors):.3f}%diag  |  "
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
