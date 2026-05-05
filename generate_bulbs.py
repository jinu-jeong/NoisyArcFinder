"""
generate_bulbs.py
─────────────────
3 bulb + glass reflection ghost composite image generator

Generation rules:
  - bulb(primary)  : bright and sharp gaussian light source (light blur applied).
                     A bulb is either a full circle, or a linearly-cut "moon" shape
                     (random angle, cut depth ≤ half — at least half of the disk remains).
                     Centroid GT is always the center of the underlying full circle,
                     so the model can fit the visible arc and recover the center.
  - ghosts         : per image, the number of ghost reflections is uniformly chosen
                     from {0, 1, 2} and applied identically to all 3 bulbs (flat glass
                     assumption). Ghosts are COLLINEAR with the primary: a single
                     base shift vector (sx, sy) is sampled per image, and ghost_k is
                     placed at primary + k · (sx, sy) for k = 1, …, n_ghosts. Each
                     ghost has its own blur and alpha (per-reflection attenuation),
                     but the displacement direction is shared. Ghost inherits each
                     bulb's shape.
  - separation     : bulbs are placed so that every bulb's group (primary + its ghosts)
                     stays at least MIN_GROUP_DIST pixels away from any point of any
                     other bulb's group. Rejection sampling enforces this.

Output:
  synthetic_data/train/  (N_TRAIN images + ground_truth.json)
  synthetic_data/test/   (N_TEST images  + ground_truth.json)
"""

import cv2
import numpy as np
import json
import os
import random

# ── Configuration ─────────────────────────────────────────
N_TRAIN    = 800
N_TEST     = 200
IMG_W      = 640
IMG_H      = 480
BASE_DIR   = "synthetic_data"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
TEST_DIR   = os.path.join(BASE_DIR, "test")
SEED       = 42
MOON_PROB  = 0.4   # per-bulb probability of being a linearly-cut "moon" shape
N_GHOSTS_CHOICES    = [0, 1, 2]   # per-image, sampled uniformly
MIN_GROUP_DIST      = 110         # px — min distance between any cross-bulb (primary or ghost) points
MAX_PLACEMENT_TRIES = 500         # rejection-sampling budget per image

random.seed(SEED)
np.random.seed(SEED)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)


# ── Helper: draw a single gaussian light source ───────────
def draw_bulb(canvas, cx, cy, radius, intensity, blur_sigma, cut=None):
    """Add a bulb to canvas (float32 H×W×3).

    Physical model:
      1. The bulb's base emitter shape is built first — a filled disk, optionally
         cut by a straight line (moon shape). The cut happens BEFORE any light
         scattering, so it represents the bulb's physical shape.
      2. Light scattering is then modeled by convolving the emitter shape with a
         gaussian PSF (sigma = radius / 2.5), producing the glow.
      3. A small extra camera blur (blur_sigma) is applied on top.

    cut: None for a full circle, or (angle, offset). Pixels with
         (X-cx)·cos(angle) + (Y-cy)·sin(angle) > offset are removed from the
         emitter. offset ∈ [0, radius] keeps ≥ half the disk.
    """
    h, w = canvas.shape[:2]
    Y, X = np.ogrid[:h, :w]
    dist2 = (X - cx) ** 2 + (Y - cy) ** 2

    # 1) Base emitter shape (cut applied to the BULB itself, before scattering)
    shape_mask = (dist2 <= radius * radius).astype(np.float32)
    if cut is not None:
        angle, offset = cut
        proj = (X - cx) * np.cos(angle) + (Y - cy) * np.sin(angle)
        shape_mask = shape_mask * (proj <= offset).astype(np.float32)

    # 2) Light scattering: emitter convolved with gaussian PSF
    psf_sigma = radius / 2.5
    k_psf = max(3, int(psf_sigma) * 2 + 1) | 1
    glow = cv2.GaussianBlur(shape_mask * intensity, (k_psf, k_psf), psf_sigma)

    # 3) Extra small camera blur
    k_blur = max(3, int(blur_sigma) * 2 + 1) | 1
    glow = cv2.GaussianBlur(glow, (k_blur, k_blur), blur_sigma)

    for c in range(3):
        canvas[:, :, c] += glow
    return canvas


# ── Helper: create ghost layer ────────────────────────────
def make_ghost_layer(primary_layer, shift_x, shift_y, blur_sigma, alpha):
    """Shift + blur + attenuate the primary layer and return ghost"""
    h, w = primary_layer.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(primary_layer, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    k = max(3, int(blur_sigma) * 2 + 1) | 1
    blurred = cv2.GaussianBlur(shifted, (k, k), blur_sigma)
    return blurred * alpha


# ── Helpers: ghost params + position sampling ─────────────
def sample_collinear_ghosts(n_ghosts):
    """Sample n_ghosts ghosts that are collinear with the primary.

    A single base displacement (base_sx, base_sy) is drawn; ghost_k is placed
    at primary + k · (base_sx, base_sy). Each ghost still has its own blur and
    alpha (subsequent reflections attenuate further). Returns [] when n_ghosts=0.
    """
    if n_ghosts == 0:
        return []
    base_sx = random.uniform(10, 25) * random.choice([-1, 1])
    base_sy = random.uniform(5,  15) * random.choice([-1, 1])
    ghosts = []
    for k in range(1, n_ghosts + 1):
        ghosts.append({
            "shift_x": k * base_sx,
            "shift_y": k * base_sy,
            "blur":    random.uniform(3, 8),
            "alpha":   random.uniform(0.25, 0.55) / k,  # k-th reflection is dimmer
        })
    return ghosts


def groups_well_separated(xs, ys, ghosts, min_dist):
    """Each bulb's group = {primary} ∪ {primary + each ghost shift}.
    Require every cross-group pair of points to be ≥ min_dist apart.
    """
    md2 = min_dist * min_dist
    groups = []
    for cx, cy in zip(xs, ys):
        pts = [(cx, cy)]
        for g in ghosts:
            pts.append((cx + g["shift_x"], cy + g["shift_y"]))
        groups.append(pts)
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for px, py in groups[i]:
                for qx, qy in groups[j]:
                    if (px - qx) ** 2 + (py - qy) ** 2 < md2:
                        return False
    return True


# ── Generate a single image ───────────────────────────────
def generate_one(idx, output_dir):
    margin     = 80
    zone_w     = (IMG_W - 2 * margin) // 3
    zone_inner = 10

    radii       = [random.randint(20, 40)      for _ in range(3)]
    intensities = [random.uniform(0.85, 1.0)   for _ in range(3)]
    pri_blurs   = [random.uniform(0.5, 1.5)    for _ in range(3)]

    # Per-bulb shape: full circle, or linearly-cut moon (≤ half cut).
    # offset ∈ [0, 0.7r]: 0 = exact half-moon, 0.7r = small sliver removed.
    cuts = []
    for r in radii:
        if random.random() < MOON_PROB:
            angle  = random.uniform(0, 2 * np.pi)
            offset = random.uniform(0.0, 0.7) * r
            cuts.append((angle, offset))
        else:
            cuts.append(None)

    # Per-image ghost count (same for every bulb). Ghosts are collinear with the
    # primary: ghost_k displacement = k · (base_sx, base_sy).
    n_ghosts = random.choice(N_GHOSTS_CHOICES)
    ghosts   = sample_collinear_ghosts(n_ghosts)

    # Rejection-sample bulb positions until all bulb groups are well separated
    xs = ys = None
    for _ in range(MAX_PLACEMENT_TRIES):
        cand_xs = []
        for i in range(3):
            x_lo = margin + i * zone_w + zone_inner
            x_hi = margin + (i + 1) * zone_w - zone_inner
            cand_xs.append(random.randint(x_lo, x_hi))
        cand_ys = [random.randint(margin, IMG_H - margin) for _ in range(3)]
        if groups_well_separated(cand_xs, cand_ys, ghosts, MIN_GROUP_DIST):
            xs, ys = cand_xs, cand_ys
            break
    if xs is None:
        # Fallback: keep last candidate (extremely unlikely with sane params)
        xs, ys = cand_xs, cand_ys

    primary_layer = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    for cx, cy, r, intens, pb, cut in zip(xs, ys, radii, intensities, pri_blurs, cuts):
        tmp = np.zeros((IMG_H, IMG_W, 3), dtype=np.float32)
        tmp = draw_bulb(tmp, cx, cy, r, intens, pb, cut=cut)
        primary_layer += tmp[:, :, 0]

    primary_rgb = np.stack([primary_layer] * 3, axis=-1)

    ghost_rgb_total = np.zeros_like(primary_rgb)
    for g in ghosts:
        ghost_rgb_total += make_ghost_layer(
            primary_rgb, g["shift_x"], g["shift_y"], g["blur"], g["alpha"],
        )

    canvas = np.clip(primary_rgb + ghost_rgb_total +
                     np.random.normal(0, 0.015, (IMG_H, IMG_W, 3)).astype(np.float32),
                     0, 1)
    img_u8 = (canvas * 255).astype(np.uint8)

    fname = f"bulbs_{idx:04d}.jpg"
    cv2.imwrite(os.path.join(output_dir, fname), img_u8)

    points = sorted(zip(xs, ys, radii, cuts), key=lambda p: p[0])
    centroids = []
    for x, y, r, cut in points:
        entry = {"x": float(x), "y": float(y), "radius": int(r)}
        if cut is None:
            entry["shape"] = "full"
            entry["cut_angle"]  = None
            entry["cut_offset"] = None
        else:
            entry["shape"] = "moon"
            entry["cut_angle"]  = float(cut[0])
            entry["cut_offset"] = float(cut[1])
        centroids.append(entry)

    return {
        "image":     fname,
        "centroids": centroids,
        "n_ghosts":  n_ghosts,
        "ghosts":    ghosts,
    }


# ── Main ──────────────────────────────────────────────────
for split, n, out_dir in [("train", N_TRAIN, TRAIN_DIR), ("test", N_TEST, TEST_DIR)]:
    print(f"\n[{split}] generating {n} images → {out_dir}/")
    metadata = []
    for idx in range(n):
        meta = generate_one(idx, out_dir)
        metadata.append(meta)
        shapes = "".join("M" if c["shape"] == "moon" else "F" for c in meta["centroids"])
        print(f"  [{idx+1:4d}/{n}] {meta['image']}  "
              f"n_ghosts={meta['n_ghosts']}  shapes={shapes}")

    gt_path = os.path.join(out_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  GT saved: {gt_path}")

print(f"\nDone! train={N_TRAIN} images / test={N_TEST} images")
