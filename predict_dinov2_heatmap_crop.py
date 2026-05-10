"""
predict_dinov2_heatmap_crop.py
──────────────────────────────
Load checkpoints/dinov2_heatmap_crop_best.pt and predict centroids for all
train/val/test images using the two-stage pipeline from
train_dinov2_heatmap_crop.py:

  Stage 1 — Gaussian-blur the full image, run greedy NMS, take 3 peaks.
  Stage 2 — For each peak, crop 200×200 around it, feed to DINOv2 head,
            decode the 64×64 multi-peak heatmap (NMS + threshold), map back
            to original-image coords.  Each crop yields 1 primary + up to 2
            ghost reflections.

Saves visualizations to prediction_result_dinov2_crop/{train,val,test}/.

Per-image overlay (BGR colors below; on screen these read as r/g/b):
  GT (circle):
    - Red    : primary
    - Green  : 1st ghost (k=1)
    - Blue   : 2nd ghost (k=2)
  Prediction (thin cross, same color scheme):
    - Red    : predicted primary
    - Green  : predicted 1st ghost
    - Blue   : predicted 2nd ghost
  Top-left text : abs / rel error for primaries and ghosts.
"""

import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train_dinov2_heatmap_crop import (
    CKPT, CROP_SIZE, DEVICE, DINO_INPUT, HEATMAP_OUT,
    IMAGENET_MEAN, IMAGENET_STD,
    DinoV2HeatmapCrop, crop_around, crop_hm_to_orig,
    decode_crop_multi_peak, detect_blurred_peaks,
)

OUT_DIR = "prediction_result_dinov2_crop"
SPLITS  = [
    ("train", "synthetic_data/train", os.path.join(OUT_DIR, "train")),
    ("val",   "synthetic_data/val",   os.path.join(OUT_DIR, "val")),
    ("test",  "synthetic_data/test",  os.path.join(OUT_DIR, "test")),
]

REF_W = 640.0

# Per-role colors (BGR, but render on screen as the listed RGB colors).
COLOR_PRIMARY = (0,   0, 255)    # red
COLOR_GHOST1  = (0, 255,   0)    # green
COLOR_GHOST2  = (255, 0,   0)    # blue


# ── Drawing helpers ───────────────────────────────────────
def draw_circle(img, cx, cy, radius, color, thickness):
    cv2.circle(img,
               (int(round(cx)), int(round(cy))),
               max(1, int(round(radius))),
               color, thickness)


def draw_x(img, cx, cy, size, color, thickness):
    icx, icy = int(round(cx)), int(round(cy))
    cv2.line(img, (icx - size, icy - size), (icx + size, icy + size),
             color, thickness, cv2.LINE_AA)
    cv2.line(img, (icx + size, icy - size), (icx - size, icy + size),
             color, thickness, cv2.LINE_AA)


# ── Inference for a single image ──────────────────────────
@torch.no_grad()
def predict_image(model, img_bgr, transform):
    """Run the two-stage pipeline.

    Returns
    -------
    peaks_orig      : (3, 2)  Stage-1 blurred peaks (= crop centers), sorted by x
    primaries_orig  : (3, 2)  predicted primary per crop, sorted by x
    ghosts_per_crop : list[3] of (G, 2) arrays — predicted ghost positions per
                      crop in original-image coords (G ∈ {0, 1, 2}), ordered
                      by distance from the predicted primary.
    """
    peaks = detect_blurred_peaks(img_bgr)         # (3, 2) sorted by x

    crop_tensors = []
    origins      = []
    for p in peaks:
        crop_bgr, origin = crop_around(img_bgr, p, CROP_SIZE)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_tensors.append(transform(Image.fromarray(crop_rgb)))
        origins.append(origin)

    batch   = torch.stack(crop_tensors).to(DEVICE)
    hm      = model(batch)                                          # (3, 1, 64, 64)
    peaks_per_crop = decode_crop_multi_peak(hm)                     # list[3] (P, 3)

    scale_crop = CROP_SIZE / HEATMAP_OUT
    primaries_orig = np.zeros((3, 2), dtype=np.float32)
    ghosts_per_crop = []
    cx_c = HEATMAP_OUT / 2.0
    cy_c = HEATMAP_OUT / 2.0

    for j, pks in enumerate(peaks_per_crop):
        ox, oy = float(origins[j][0]), float(origins[j][1])
        if pks.shape[0] == 0:
            primaries_orig[j] = peaks[j]                            # fallback
            ghosts_per_crop.append(np.zeros((0, 2), dtype=np.float32))
            continue
        d2 = (pks[:, 0] - cx_c) ** 2 + (pks[:, 1] - cy_c) ** 2
        prim_i = int(d2.argmin())
        prim_hm = pks[prim_i, :2]
        primaries_orig[j, 0] = prim_hm[0] * scale_crop + ox
        primaries_orig[j, 1] = prim_hm[1] * scale_crop + oy

        ghosts_hm = np.delete(pks[:, :2], prim_i, axis=0)            # (G, 2)
        # order ghosts by distance from primary so they're comparable to k=1, k=2
        if ghosts_hm.shape[0] > 1:
            d2g = ((ghosts_hm - prim_hm) ** 2).sum(axis=1)
            ghosts_hm = ghosts_hm[np.argsort(d2g)]
        ghosts_orig = ghosts_hm * scale_crop + np.array([ox, oy], dtype=np.float32)
        ghosts_per_crop.append(ghosts_orig.astype(np.float32))

    return peaks, primaries_orig, ghosts_per_crop


# ── GT helpers ────────────────────────────────────────────
def gt_ghost_positions(rec):
    """Return list[3] of (n_ghosts, 2) arrays — GT ghost positions per primary,
    sorted by primary x to match predicted-primary ordering.

    Each centroid in ground_truth.json carries its OWN ghost list (shifts can
    differ between the 3 bulb-complexes), so we read shifts off the centroid
    record rather than off a top-level shared list.
    """
    sorted_centroids = sorted(rec["centroids"], key=lambda c: c["x"])
    out = []
    for c in sorted_centroids:
        cx, cy = float(c["x"]), float(c["y"])
        ghosts = c.get("ghosts") or []
        gh = np.array(
            [(cx + g["shift_x"], cy + g["shift_y"]) for g in ghosts],
            dtype=np.float32,
        ).reshape(-1, 2)
        out.append(gh)
    return out


def match_ghost_errors(pred_ghosts, gt_ghosts):
    """Return per-bulb mean L2 error for ghosts when both pred and GT are
    non-empty, else None.  pred_ghosts and gt_ghosts are ordered the same way
    (by distance from primary / by k), so positional matching is enough.
    """
    if pred_ghosts.shape[0] == 0 or gt_ghosts.shape[0] == 0:
        return None
    n = min(pred_ghosts.shape[0], gt_ghosts.shape[0])
    d = np.linalg.norm(pred_ghosts[:n] - gt_ghosts[:n], axis=1)
    return float(d.mean())


# ── Per-split inference + visualization ───────────────────
def predict_split(model, split_name, data_dir, out_dir, transform):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(data_dir, "ground_truth.json")) as f:
        records = json.load(f)

    total_prim_abs = 0.0
    total_prim_rel = 0.0
    prim_abs_errs  = []

    n_ghost_imgs   = 0
    total_ghost_abs = 0.0
    ghost_abs_errs  = []

    for rec in records:
        img_path = os.path.join(data_dir, rec["image"])
        img_bgr  = cv2.imread(img_path)
        H, W     = img_bgr.shape[:2]
        diag     = (W ** 2 + H ** 2) ** 0.5

        peaks, primaries_orig, ghosts_per_crop = predict_image(
            model, img_bgr, transform
        )

        # ── Primary error (sorted by x in both pred and GT) ───────
        sorted_gt_prim = sorted(
            [(c["x"], c["y"]) for c in rec["centroids"]], key=lambda p: p[0]
        )
        gt_prim = np.array(sorted_gt_prim, dtype=np.float32)        # (3, 2)
        prim_abs = float(np.linalg.norm(primaries_orig - gt_prim, axis=1).mean())
        prim_rel = prim_abs / diag * 100
        total_prim_abs += prim_abs
        total_prim_rel += prim_rel
        prim_abs_errs.append(prim_abs)

        # ── Ghost error: only over images with n_ghosts > 0 ───────
        gt_ghosts_per_bulb = gt_ghost_positions(rec)
        n_ghosts = rec.get("n_ghosts", 0)
        ghost_abs_per_bulb = []
        if n_ghosts > 0:
            for pg, gg in zip(ghosts_per_crop, gt_ghosts_per_bulb):
                e = match_ghost_errors(pg, gg)
                if e is not None:
                    ghost_abs_per_bulb.append(e)
            if ghost_abs_per_bulb:
                ghost_abs = sum(ghost_abs_per_bulb) / len(ghost_abs_per_bulb)
                total_ghost_abs += ghost_abs
                ghost_abs_errs.append(ghost_abs)
                n_ghost_imgs += 1

        # ── Draw ──────────────────────────────────────────
        vis   = img_bgr.copy()
        scale = W / REF_W

        gt_thick     = max(1, int(round(1.5 * scale)))    # circle outline
        pred_thick   = max(1, int(round(0.5 * scale)))    # thin cross
        x_size       = max(4, int(round(9 * scale)))
        ghost_x_size = max(3, int(round(7 * scale)))
        gt_ghost_r   = max(3, int(round(8 * scale)))

        # GT primary: red circle (no crosshair)
        for c in sorted(rec["centroids"], key=lambda c: c["x"]):
            draw_circle(vis, c["x"], c["y"],
                        radius=c.get("radius", 30),
                        color=COLOR_PRIMARY,
                        thickness=gt_thick)

        # GT ghost circles: green for k=1, blue for k=2
        for gg in gt_ghosts_per_bulb:
            for k, (gx, gy) in enumerate(gg.tolist()):
                color = COLOR_GHOST1 if k == 0 else COLOR_GHOST2
                draw_circle(vis, gx, gy, gt_ghost_r, color, gt_thick)

        # Predicted primary: red thin cross
        for px, py in primaries_orig.tolist():
            draw_x(vis, px, py, size=x_size,
                   color=COLOR_PRIMARY, thickness=pred_thick)

        # Predicted ghosts: green (k=1) / blue (k=2) thin crosses.
        # ghosts_per_crop[j] is already ordered by distance from the predicted
        # primary, so [0] is the 1st-ghost prediction, [1] the 2nd-ghost.
        for pg in ghosts_per_crop:
            for k, (gx, gy) in enumerate(pg.tolist()):
                color = COLOR_GHOST1 if k == 0 else COLOR_GHOST2
                draw_x(vis, gx, gy, size=ghost_x_size,
                       color=color, thickness=pred_thick)

        # Error text
        font_scale = 0.80 * scale
        font_thick = max(1, int(round(1.5 * scale)))
        margin_px  = max(6, int(round(8  * scale)))
        line_h     = max(16, int(round(30 * scale)))

        cv2.putText(vis, f"primary  abs={prim_abs:.1f}px  rel={prim_rel:.3f}%diag",
                    (margin_px, line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 220, 255), font_thick, cv2.LINE_AA)
        if n_ghosts > 0 and ghost_abs_per_bulb:
            ghost_abs = sum(ghost_abs_per_bulb) / len(ghost_abs_per_bulb)
            cv2.putText(vis,
                        f"ghosts  abs={ghost_abs:.1f}px  (n={n_ghosts})",
                        (margin_px, line_h * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 220, 255), font_thick, cv2.LINE_AA)
        else:
            cv2.putText(vis, f"ghosts  n={n_ghosts}",
                        (margin_px, line_h * 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 220, 255), font_thick, cv2.LINE_AA)

        cv2.imwrite(os.path.join(out_dir, rec["image"]), vis)

    n = len(records)
    msg = (f"[{split_name}]  {n} images  |  "
           f"primary mean abs = {total_prim_abs / n:.2f}px  "
           f"max = {max(prim_abs_errs):.1f}px  "
           f"mean rel = {total_prim_rel / n:.3f}%diag")
    if n_ghost_imgs > 0:
        msg += (f"  |  ghosts on {n_ghost_imgs} imgs: "
                f"mean abs = {total_ghost_abs / n_ghost_imgs:.2f}px  "
                f"max = {max(ghost_abs_errs):.1f}px")
    msg += f"  |  saved → {out_dir}/"
    print(msg)


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((DINO_INPUT, DINO_INPUT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    model = DinoV2HeatmapCrop().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CKPT}  (device={DEVICE})\n")

    for split_name, data_dir, out_dir in SPLITS:
        predict_split(model, split_name, data_dir, out_dir, transform)
