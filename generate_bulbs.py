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
  synthetic_data/train/  (N_TRAIN images + ground_truth.json)  — 70 %
  synthetic_data/val/    (N_VAL   images + ground_truth.json)  — 10 %
  synthetic_data/test/   (N_TEST  images + ground_truth.json)  — 20 %
"""

import cv2
import numpy as np
import json
import os
import random
from multiprocessing import Pool
import os as _os

# ── Configuration ─────────────────────────────────────────
N = 10000
N_TRAIN = int(N * 0.7)
N_VAL   = int(N * 0.1)
N_TEST  = N - N_TRAIN - N_VAL

BASE_DIR   = "synthetic_data"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
TEST_DIR   = os.path.join(BASE_DIR, "test")
SEED       = 42

# Per-image random resolution
W_MIN, W_MAX   = 1000, 4500   # image width range (px)
AR_MIN, AR_MAX = 0.4,  0.6    # height/width aspect ratio range

MOON_PROB           = 0.4   # per-bulb probability of being a linearly-cut "moon" shape
N_GHOSTS_CHOICES    = [0, 1, 2]   # per-image, sampled uniformly
MAX_PLACEMENT_TRIES = 500         # rejection-sampling budget per image

# All spatial parameters are derived proportionally inside generate_one:
#   margin        ≈ 12.5 % of img_w
#   zone_inner    ≈  1.6 % of img_w
#   bulb radius   ≈  3 – 6 % of img_w
#   min_group_dist≈ 17   % of img_w
#   ghost shift   ≈  1.6 – 3.9 % of img_w (x),  1.0 – 3.1 % of img_h (y)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR,   exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)


# ── Helper: draw a single gaussian light source ───────────
def draw_bulb(canvas, cx, cy, radius, intensity, blur_sigma, cut=None, grid=None):
    """Add a bulb to canvas (float32 H×W×3).

    Physical model:
      1. The bulb's base emitter shape is built first — a filled disk, optionally
         cut by a straight line (moon shape). The cut happens BEFORE any light
         scattering, so it represents the bulb's physical shape.
      2. Light scattering is then modeled by convolving the emitter shape with a
         gaussian PSF (sigma = radius / 2.5), producing the glow.
      3. A small extra camera blur (blur_sigma) is applied on top.

    cut:  None for a full circle, or (angle, offset). Pixels with
          (X-cx)·cos(angle) + (Y-cy)·sin(angle) > offset are removed from the
          emitter. offset ∈ [0, radius] keeps ≥ half the disk.
    grid: optional pre-built (Y, X) ogrid tuple to avoid reallocation per bulb.
    """
    h, w = canvas.shape[:2]
    if grid is None:
        Y, X = np.ogrid[:h, :w]
    else:
        Y, X = grid
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
def sample_collinear_ghosts(n_ghosts, img_w, img_h):
    """Sample n_ghosts ghosts that are collinear with the primary.

    All displacements and blur values are proportional to the image dimensions
    so that the ghost geometry looks consistent regardless of resolution.
    Returns [] when n_ghosts=0.
    """
    if n_ghosts == 0:
        return []
    base_sx = random.uniform(img_w * 0.016, img_w * 0.039) * random.choice([-1, 1])
    base_sy = random.uniform(img_h * 0.010, img_h * 0.031) * random.choice([-1, 1])
    blur_scale = img_w / 640.0   # keep blur visually consistent vs. 640 px reference
    ghosts = []
    for k in range(1, n_ghosts + 1):
        ghosts.append({
            "shift_x": k * base_sx,
            "shift_y": k * base_sy,
            "blur":    random.uniform(3, 8) * blur_scale,
            "alpha":   random.uniform(0.25, 0.55) / k,
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
    # ── Random image dimensions ────────────────────────────
    img_w = random.randint(W_MIN, W_MAX)
    img_h = int(img_w * random.uniform(AR_MIN, AR_MAX))

    # ── Proportional spatial parameters ───────────────────
    margin         = int(img_w * 0.125)
    zone_inner     = int(img_w * 0.016)
    zone_w         = (img_w - 2 * margin) // 3
    min_group_dist = int(img_w * 0.17)

    r_min = max(4, int(img_w * 0.030))
    r_max = max(r_min + 1, int(img_w * 0.060))

    radii       = [random.randint(r_min, r_max) for _ in range(3)]
    intensity   = random.uniform(0.3, 1.0)          # same peak brightness for all 3 bulbs
    intensities = [intensity] * 3
    pri_blurs   = [random.uniform(0.02, 0.06) * r for r in radii]  # scales with radius

    # Per-bulb shape: full circle, or linearly-cut moon (≤ half cut).
    cuts = []
    for r in radii:
        if random.random() < MOON_PROB:
            angle  = random.uniform(0, 2 * np.pi)
            offset = random.uniform(0.0, 0.7) * r
            cuts.append((angle, offset))
        else:
            cuts.append(None)

    # Per-image ghost count; collinear with primary, proportional shifts.
    n_ghosts = random.choice(N_GHOSTS_CHOICES)
    ghosts   = sample_collinear_ghosts(n_ghosts, img_w, img_h)

    # Rejection-sample bulb positions until all groups are well separated.
    xs = ys = None
    for _ in range(MAX_PLACEMENT_TRIES):
        cand_xs = []
        for i in range(3):
            x_lo = margin + i * zone_w + zone_inner
            x_hi = margin + (i + 1) * zone_w - zone_inner
            cand_xs.append(random.randint(x_lo, x_hi))
        cand_ys = [random.randint(margin, img_h - margin) for _ in range(3)]
        if groups_well_separated(cand_xs, cand_ys, ghosts, min_group_dist):
            xs, ys = cand_xs, cand_ys
            break
    if xs is None:
        xs, ys = cand_xs, cand_ys  # fallback (extremely unlikely)

    grid = np.ogrid[:img_h, :img_w]   # build once, reuse for every bulb
    primary_layer = np.zeros((img_h, img_w), dtype=np.float32)
    for cx, cy, r, intens, pb, cut in zip(xs, ys, radii, intensities, pri_blurs, cuts):
        tmp = np.zeros((img_h, img_w, 3), dtype=np.float32)
        tmp = draw_bulb(tmp, cx, cy, r, intens, pb, cut=cut, grid=grid)
        primary_layer += tmp[:, :, 0]

    primary_rgb = np.stack([primary_layer] * 3, axis=-1)

    ghost_rgb_total = np.zeros_like(primary_rgb)
    for g in ghosts:
        ghost_rgb_total += make_ghost_layer(
            primary_rgb, g["shift_x"], g["shift_y"], g["blur"], g["alpha"],
        )

    canvas = np.clip(primary_rgb + ghost_rgb_total +
                     np.random.normal(0, 0.015, (img_h, img_w, 3)).astype(np.float32),
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
        "img_w":     img_w,
        "img_h":     img_h,
        "intensity": intensity,
        "centroids": centroids,
        "n_ghosts":  n_ghosts,
        "ghosts":    ghosts,
    }


# ── Worker (must be top-level for multiprocessing pickling) ──
def _worker(args):
    """Set single-threaded BLAS/OpenMP per worker to avoid over-subscription,
    and seed each worker with (SEED + idx) for reproducibility."""
    _os.environ.setdefault("OMP_NUM_THREADS", "1")
    _os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    idx, out_dir = args
    random.seed(SEED + idx)
    np.random.seed(SEED + idx)
    return generate_one(idx, out_dir)


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    n_workers = max(1, _os.cpu_count() or 1)
    print(f"Using {n_workers} worker(s)\n")

    splits = [
        ("train", N_TRAIN, TRAIN_DIR),
        ("val",   N_VAL,   VAL_DIR),
        ("test",  N_TEST,  TEST_DIR),
    ]
    for split, n, out_dir in splits:
        print(f"[{split}] generating {n} images → {out_dir}/")
        args = [(idx, out_dir) for idx in range(n)]

        with Pool(processes=n_workers) as pool:
            # imap_unordered for progress; results collected in order via dict
            results = {}
            for meta in pool.imap_unordered(_worker, args, chunksize=4):
                idx = int(meta["image"].split("_")[1].split(".")[0])
                results[idx] = meta
                shapes = "".join("M" if c["shape"] == "moon" else "F"
                                 for c in meta["centroids"])
                print(f"  [{idx+1:4d}/{n}] {meta['image']}  "
                      f"{meta['img_w']}x{meta['img_h']}  "
                      f"n_ghosts={meta['n_ghosts']}  shapes={shapes}")

        metadata = [results[i] for i in range(n)]
        gt_path = os.path.join(out_dir, "ground_truth.json")
        with open(gt_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  GT saved: {gt_path}\n")

    print(f"Done! train={N_TRAIN} / val={N_VAL} / test={N_TEST} images")
