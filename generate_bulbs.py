"""
generate_bulbs.py
─────────────────
3 bulb + glass reflection ghost composite image generator

Generation rules:
  - bulb(primary)  : bright and sharp gaussian light source (light blur applied).
                     A bulb is either a full circle, or a linearly-cut "moon"
                     shape with random angle and cut depth. The cut can extend
                     past the centre (crescent), so the centroid GT — always
                     the centre of the underlying full circle — may land in
                     the cut-away dark region. The model must fit the visible
                     arc to recover the implicit centre even when it is off
                     the bright pixels.
  - ghosts         : per image, the number of ghost reflections is uniformly chosen
                     from {0, 1, 2}. Each of the 3 bulb-complexes gets its OWN
                     base shift vector (sx, sy) drawn from [0, MAX] in each axis
                     with an independent random sign — so within one image, the
                     left bulb's ghost separation can be small while the middle
                     bulb's is large (or vice versa, or anything in between).
                     Ghost_k of bulb b is placed at primary_b + k · (sx_b, sy_b).
                     Each ghost is a SCALED-DOWN copy of the primary's full shape
                     (cut mask, diffraction spikes, glow), redrawn at smaller
                     radius so the star pattern is preserved while the apparent
                     size shrinks with order. Per-image multiplicative steps make
                     sizes and brightnesses descend monotonically (k=2 smaller
                     and fainter than k=1) — sometimes mildly, sometimes
                     drastically — while a shared base scattering σ keeps every
                     ghost's diffuseness similar across the 3 bulbs. Ghosts are
                     comparable in brightness to the primary (alpha_1 ∈ 0.85–0.97)
                     but always strictly fainter and smaller.
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
N = 5000
N_TRAIN = int(N * 0.7)
N_VAL   = int(N * 0.1)
N_TEST  = N - N_TRAIN - N_VAL

BASE_DIR   = "synthetic_data"
TRAIN_DIR  = os.path.join(BASE_DIR, "train")
VAL_DIR    = os.path.join(BASE_DIR, "val")
TEST_DIR   = os.path.join(BASE_DIR, "test")
SEED       = 42

# Per-image resolution: base 3840×2158, scaled by a Gaussian multiplier
IMG_BASE_W      = 3840          # base image width  (px)
IMG_BASE_H      = 2158          # base image height (px)
IMG_SCALE_MU    = 1.0           # Gaussian mean for the scale multiplier
IMG_SCALE_SIGMA = 0.15          # Gaussian std  for the scale multiplier
IMG_SCALE_MIN   = 0.5           # hard clamp – never smaller than 50 % of base
IMG_SCALE_MAX   = 1.5           # hard clamp – never larger  than 150 % of base

# Per-bulb radius: base 20 px, scaled by a Gaussian multiplier
BULB_BASE_R     = 20            # base bulb radius (px)
BULB_SCALE_MU   = 1.0           # Gaussian mean for the radius multiplier
BULB_SCALE_SIGMA = 0.2          # Gaussian std  for the radius multiplier
BULB_SCALE_MIN  = 0.25          # hard clamp – minimum radius = 5 px
BULB_SCALE_MAX  = 2.0           # hard clamp – maximum radius = 40 px

MOON_PROB           = 0.4   # per-bulb probability of being a linearly-cut "moon" shape
N_GHOSTS_CHOICES    = [2]         # always 2 ghosts per image
MAX_PLACEMENT_TRIES = 500         # rejection-sampling budget per image

# Diffraction spikes
# 'length' is the Gaussian σ along the spike axis (px); visual half-extent ≈ 3σ.
# Max 10 × diameter total = 20r → 3σ ≤ 10r → σ_max = 3.3r  → LEN_MAX = 3.3.
# Amplitude must stay below the glow at the disk edge (≈ 0.5 × intensity),
# so AMP_MAX is capped at 0.25 to keep spikes visually dimmer than the bulb.
SPIKE_PROB          = 0.0         # probability that a bulb has diffraction spikes (0 = disabled)
SPIKE_N_PAIRS       = [5, 6, 7, 8, 9, 10]  # 5–10 pairs → 10–20 spokes total
SPIKE_LEN_MU        = 2.0         # Gaussian mean  for spike σ (× radius)
SPIKE_LEN_SIGMA     = 0.5         # Gaussian std   for spike σ (× radius)
SPIKE_LEN_MIN       = 0.8         # minimum σ multiplier
SPIKE_LEN_MAX       = 3.3         # maximum σ multiplier → visual half-extent ≤ 10r = 5 diameters
SPIKE_WIDTH_FRAC    = 0.15        # perpendicular σ as a fraction of radius (min 0.8 px)
SPIKE_AMP_MU        = 0.30        # Gaussian mean  for spike peak amplitude (× intensity)
SPIKE_AMP_SIGMA     = 0.08        # Gaussian std   for spike peak amplitude
SPIKE_AMP_MIN       = 0.12        # minimum amplitude fraction
SPIKE_AMP_MAX       = 0.50        # maximum amplitude fraction (max() composition keeps spike < core peak)

# All spatial parameters are derived proportionally inside generate_one:
#   margin        ≈ 12.5 % of img_w
#   zone_inner    ≈  1.6 % of img_w
#   bulb radius   ≈  base 20 px × Gaussian(1.0, 0.2), clamped to [5, 40] px
#   min_group_dist≈ 17   % of img_w
#   ghost shift   ≈  1.6 – 3.9 % of img_w (x),  1.0 – 3.1 % of img_h (y)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR,   exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)


# ── Helper: draw a single gaussian light source ───────────
def draw_bulb(canvas, cx, cy, radius, intensity, blur_sigma, cut=None, spikes=None):
    """Add a bulb to canvas (float32 H×W×3) using a tight bounding box.

    All array operations are performed on a small local patch rather than the
    full image.  The patch is pasted back via in-place addition, keeping the
    memory and compute cost proportional to the bulb/spike size, not the image.

    cut:    None for a full circle, or (angle, offset). Pixels with
            (X-cx)·cos(angle) + (Y-cy)·sin(angle) > offset are removed from the
            emitter. offset ∈ [0, radius] keeps ≥ half the disk.
    spikes: None, or dict with keys
              n_pairs    – number of spike pairs (2 → 4 spikes, 3 → 6 spikes)
              base_angle – rotation of the first pair (radians)
              lengths    – per-pair Gaussian σ along the spike axis (list, px);
                           visual extent of pair i ≈ ±3 × lengths[i]
              width_sigma– perpendicular Gaussian σ (pixels)
              amplitude  – peak value (already scaled by intensity)
    """
    h, w = canvas.shape[:2]

    # ── Bounding box ──────────────────────────────────────────────────────────
    psf_sigma = radius / 2.5
    glow_pad  = int(np.ceil(radius + 3 * (psf_sigma + blur_sigma))) + 2

    # Axis-aligned bounding box of every spike ellipse (3σ extent each axis)
    spike_pad_x = spike_pad_y = 0
    if spikes is not None:
        W3 = 3 * spikes["width_sigma"]
        for i in range(spikes["n_pairs"]):
            theta = spikes["base_angle"] + i * (np.pi / spikes["n_pairs"])
            L3    = 3 * spikes["lengths"][i]
            ca, sa = abs(np.cos(theta)), abs(np.sin(theta))
            spike_pad_x = max(spike_pad_x, int(np.ceil(ca * L3 + sa * W3)))
            spike_pad_y = max(spike_pad_y, int(np.ceil(sa * L3 + ca * W3)))

    pad_x = max(glow_pad, spike_pad_x)
    pad_y = max(glow_pad, spike_pad_y)

    cx_i, cy_i = int(round(cx)), int(round(cy))
    x0 = max(0, cx_i - pad_x);  x1 = min(w, cx_i + pad_x + 1)
    y0 = max(0, cy_i - pad_y);  y1 = min(h, cy_i + pad_y + 1)
    if x1 <= x0 or y1 <= y0:
        return canvas

    # ── Local coordinate arrays (image-absolute, so cx/cy math stays correct) ─
    lX = np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]   # (1, lw)
    lY = np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]   # (lh, 1)
    dx    = lX - cx
    dy    = lY - cy
    dist2 = dx ** 2 + dy ** 2

    # 1) Base emitter shape (cut applied before scattering)
    shape_mask = (dist2 <= radius * radius).astype(np.float32)
    if cut is not None:
        angle, offset = cut
        proj = dx * np.cos(angle) + dy * np.sin(angle)
        shape_mask *= (proj <= offset).astype(np.float32)

    # 2) Light scattering: emitter convolved with gaussian PSF
    k_psf = max(3, int(psf_sigma) * 2 + 1) | 1
    glow  = cv2.GaussianBlur(shape_mask * intensity, (k_psf, k_psf), psf_sigma)

    # 3) Extra small camera blur
    k_blur = max(3, int(blur_sigma) * 2 + 1) | 1
    glow   = cv2.GaussianBlur(glow, (k_blur, k_blur), blur_sigma)

    # 4) Diffraction spikes – rotated 2-D Gaussians, one per spoke direction.
    # Compose with max(), not +: the spike is the same emitted light redirected
    # by diffraction, so it must never make a pixel brighter than the glow peak.
    if spikes is not None:
        lengths = spikes["lengths"]
        w_sigma = spikes["width_sigma"]
        amp     = spikes["amplitude"]
        # No interior suppression: the spike runs continuously through the bulb
        # so a cut (moon) bulb still shows a connected spike line. The glow
        # always dominates the spike inside an intact disk because
        # SPIKE_AMP_MAX < glow_peak, so np.maximum keeps the core brightest.
        spike_layer = np.zeros_like(glow)
        for i in range(spikes["n_pairs"]):
            theta = spikes["base_angle"] + i * (np.pi / spikes["n_pairs"])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            u =  dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            spike = (amp
                     * np.exp(-0.5 * (u / lengths[i]) ** 2)
                     * np.exp(-0.5 * (v / w_sigma) ** 2))
            np.maximum(spike_layer, spike, out=spike_layer)
        glow = np.maximum(glow, spike_layer)

    # ── Paste local patch back into canvas ────────────────────────────────────
    canvas[y0:y1, x0:x1, 0] += glow
    canvas[y0:y1, x0:x1, 1] += glow
    canvas[y0:y1, x0:x1, 2] += glow
    return canvas


# ── Helper: scale per-bulb draw params for a ghost order ──
def _scale_bulb_params(radii, pri_blurs, cuts, spike_configs, size_scale):
    """Return (radii, pri_blurs, cuts, spikes) with all spatial sizes shrunk
    by `size_scale`.  Spike *direction*/*count* are preserved, only their
    length and width are scaled — this keeps the diffraction/scattering
    pattern identical across primary and every ghost order, so a viewer
    sees the same star but smaller.
    """
    radii_g = [max(2, int(round(r * size_scale))) for r in radii]
    blurs_g = [pb * size_scale for pb in pri_blurs]
    cuts_g  = [None if c is None else (c[0], c[1] * size_scale) for c in cuts]
    spikes_g = []
    for spk in spike_configs:
        if spk is None:
            spikes_g.append(None)
            continue
        spikes_g.append({
            "n_pairs":     spk["n_pairs"],         # same star pattern
            "base_angle":  spk["base_angle"],      # same orientation
            "lengths":     [L * size_scale for L in spk["lengths"]],
            "width_sigma": max(0.5, spk["width_sigma"] * size_scale),
            "amplitude":   spk["amplitude"],       # alpha controls overall fade
        })
    return radii_g, blurs_g, cuts_g, spikes_g


# ── Helper: render a ghost layer for ONE bulb (scaled → shift → blur → α) ─
def render_single_ghost_layer(canvas_h, canvas_w,
                              cx, cy, radius, intensity, pri_blur, cut, spike_config,
                              size_scale, shift_x, shift_y, blur_sigma, alpha):
    """Draw a single scaled-down copy of one bulb at (cx, cy), translate by
    (shift_x, shift_y), Gaussian-blur with `blur_sigma`, then attenuate by
    `alpha`.  Each (bulb, ghost-order) pair is rendered independently so each
    bulb-complex can have its own primary→ghost shift."""
    rs, pbs, cs, sps = _scale_bulb_params(
        [radius], [pri_blur], [cut], [spike_config], size_scale,
    )
    layer = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    draw_bulb(layer, cx, cy, rs[0], intensity, pbs[0], cut=cs[0], spikes=sps[0])

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(layer, M, (canvas_w, canvas_h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
    k = max(3, int(blur_sigma) * 2 + 1) | 1
    blurred = cv2.GaussianBlur(shifted, (k, k), blur_sigma)
    return blurred * alpha


# Per-bulb ghost shifts are capped in ABSOLUTE pixels (not as a fraction of
# image size) so the maximum primary→2nd-ghost distance is bounded regardless
# of img_w / img_h.  At 80 px per axis, the 2nd ghost is at most 2·80 = 160 px
# in each axis from its primary, fitting comfortably inside CROP_SIZE = 500.
GHOST_MAX_SHIFT_PX = 80    # |base_sx|, |base_sy| ∈ [0, GHOST_MAX_SHIFT_PX]


# ── Helpers: ghost params + position sampling ─────────────
def sample_collinear_ghosts(n_ghosts, img_w, img_h, radii):
    """Sample n_ghosts collinear ghosts for each of the 3 bulbs.

    Returns a list of length 3 (one entry per bulb). Each entry is a list of
    n_ghosts ghost dicts ordered by reflection order k = 1 … n_ghosts.

    Per-bulb (independent across the 3 bulbs in one image):
      • shift  : each bulb gets its own (sx, sy) drawn uniformly from the
                 closed range [0, MAX] in each axis, with an independent
                 random sign.  Because the lower bound is 0, two of three
                 bulb-complexes can be tightly stacked while the third is
                 widely separated (or anything in between).
                 ghost_k of bulb b is at primary_b + k · (sx_b, sy_b).

    Per-image (shared across the 3 bulbs to keep "scattering pattern similar"):
      • size schedule  : size_1 ∈ [0.55, 0.85], size_step ∈ [0.45, 0.95].
                         Every ghost is strictly smaller than the primary and
                         each subsequent order is smaller still.
      • alpha schedule : alpha_1 ∈ [0.85, 0.97], alpha_step ∈ [0.70, 0.92].
                         Comparable to primary, but always darker.
      • blur character : σ_g/radius ∈ [0.20, 0.45] shared across the image,
                         multiplied by each ghost's own size_scale so a small
                         ghost stays proportionally sharp.
    """
    if n_ghosts == 0:
        return [[] for _ in range(3)]

    mean_r          = float(sum(radii)) / len(radii)
    base_blur_ratio = random.uniform(0.20, 0.45)     # σ_g / radius
    base_blur       = base_blur_ratio * mean_r       # absolute σ at primary size
    blur_jitter     = 0.10 * base_blur

    alpha_1    = random.uniform(0.85, 0.97)
    alpha_step = random.uniform(0.70, 0.92)

    size_1     = random.uniform(0.55, 0.85)
    size_step  = random.uniform(0.45, 0.95)

    ghosts_per_bulb = []
    for _b in range(3):
        # Per-bulb-complex shift — minimum 0 so a ghost can sit on top of
        # its primary, max ≈ current upper bound.  Sign is independent per
        # axis and per bulb.
        sx_b = random.uniform(0, GHOST_MAX_SHIFT_PX) * random.choice([-1, 1])
        sy_b = random.uniform(0, GHOST_MAX_SHIFT_PX) * random.choice([-1, 1])

        bulb_ghosts = []
        for k in range(1, n_ghosts + 1):
            size  = size_1  * (size_step  ** (k - 1))
            alpha = alpha_1 * (alpha_step ** (k - 1))
            blur  = max(0.5,
                        base_blur * size + random.uniform(-blur_jitter, blur_jitter))
            bulb_ghosts.append({
                "shift_x": k * sx_b,
                "shift_y": k * sy_b,
                "blur":    blur,
                "alpha":   alpha,
                "size":    size,
            })
        ghosts_per_bulb.append(bulb_ghosts)
    return ghosts_per_bulb


def groups_well_separated(xs, ys, ghosts_per_bulb, min_dist):
    """Each bulb's group = {primary} ∪ {primary + each of *its own* ghost shifts}.
    Require every cross-group pair of points to be ≥ min_dist apart.
    """
    md2 = min_dist * min_dist
    groups = []
    for cx, cy, bulb_ghosts in zip(xs, ys, ghosts_per_bulb):
        pts = [(cx, cy)]
        for g in bulb_ghosts:
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
    # ── Image dimensions: base 3840×2158 × Gaussian multiplier ───
    img_scale = np.clip(random.gauss(IMG_SCALE_MU, IMG_SCALE_SIGMA),
                        IMG_SCALE_MIN, IMG_SCALE_MAX)
    img_w = max(64, int(IMG_BASE_W * img_scale))
    img_h = max(64, int(IMG_BASE_H * img_scale))

    # ── Proportional spatial parameters ───────────────────
    margin         = int(img_w * 0.125)
    zone_inner     = int(img_w * 0.016)
    zone_w         = (img_w - 2 * margin) // 3
    min_group_dist = int(img_w * 0.17)

    # ── Bulb radii: base 20 px × per-bulb Gaussian multiplier ────
    radii = []
    for _ in range(3):
        scale = np.clip(random.gauss(BULB_SCALE_MU, BULB_SCALE_SIGMA),
                        BULB_SCALE_MIN, BULB_SCALE_MAX)
        radii.append(max(5, int(BULB_BASE_R * scale)))
    intensity   = random.uniform(0.3, 1.0)          # same peak brightness for all 3 bulbs
    intensities = [intensity] * 3
    pri_blurs   = [random.uniform(0.02, 0.06) * r for r in radii]  # scales with radius

    # Per-bulb diffraction spikes; each spike pair gets its own random length
    spike_configs = []
    for r in radii:
        if random.random() < SPIKE_PROB:
            n_pairs  = random.choice(SPIKE_N_PAIRS)
            lengths  = [
                r * np.clip(random.gauss(SPIKE_LEN_MU, SPIKE_LEN_SIGMA),
                            SPIKE_LEN_MIN, SPIKE_LEN_MAX)
                for _ in range(n_pairs)
            ]
            amp_frac = np.clip(random.gauss(SPIKE_AMP_MU, SPIKE_AMP_SIGMA),
                               SPIKE_AMP_MIN, SPIKE_AMP_MAX)
            spike_configs.append({
                "n_pairs":     n_pairs,
                "base_angle":  random.uniform(0, np.pi),
                "lengths":     lengths,
                "width_sigma": max(0.5, r * SPIKE_WIDTH_FRAC),
                "amplitude":   intensity * amp_frac,
            })
        else:
            spike_configs.append(None)

    # Per-bulb shape: full circle, or linearly-cut moon. The cut can extend
    # past the disk centre (negative offset → crescent), in which case the
    # centroid GT sits in the cut-away dark region rather than on the arc.
    cuts = []
    for r in radii:
        if random.random() < MOON_PROB:
            angle  = random.uniform(0, 2 * np.pi)
            offset = random.uniform(-0.5, 0.7) * r
            cuts.append((angle, offset))
        else:
            cuts.append(None)

    # Per-image ghost count; ghost size/alpha schedules and blur character are
    # shared across the 3 bulbs, but each bulb-complex gets its own (sx, sy)
    # shift drawn from [0, MAX] — different separations within one image.
    n_ghosts        = random.choice(N_GHOSTS_CHOICES)
    ghosts_per_bulb = sample_collinear_ghosts(n_ghosts, img_w, img_h, radii)

    # Rejection-sample bulb positions until all groups are well separated.
    xs = ys = None
    for _ in range(MAX_PLACEMENT_TRIES):
        cand_xs = []
        for i in range(3):
            x_lo = margin + i * zone_w + zone_inner
            x_hi = margin + (i + 1) * zone_w - zone_inner
            cand_xs.append(random.randint(x_lo, x_hi))
        cand_ys = [random.randint(margin, img_h - margin) for _ in range(3)]
        if groups_well_separated(cand_xs, cand_ys, ghosts_per_bulb,
                                 min_group_dist):
            xs, ys = cand_xs, cand_ys
            break
    if xs is None:
        xs, ys = cand_xs, cand_ys  # fallback (extremely unlikely)

    primary_canvas = np.zeros((img_h, img_w, 3), dtype=np.float32)
    for cx, cy, r, intens, pb, cut, spk in zip(
            xs, ys, radii, intensities, pri_blurs, cuts, spike_configs):
        draw_bulb(primary_canvas, cx, cy, r, intens, pb, cut=cut, spikes=spk)

    # Each bulb's ghosts are rendered with that bulb's own shift.
    ghost_rgb_total = np.zeros((img_h, img_w, 3), dtype=np.float32)
    for b in range(3):
        for g in ghosts_per_bulb[b]:
            ghost_rgb_total += render_single_ghost_layer(
                img_h, img_w,
                xs[b], ys[b], radii[b], intensities[b], pri_blurs[b],
                cuts[b], spike_configs[b],
                size_scale=g["size"],
                shift_x=g["shift_x"], shift_y=g["shift_y"],
                blur_sigma=g["blur"], alpha=g["alpha"],
            )

    canvas = np.clip(primary_canvas + ghost_rgb_total +
                     np.random.normal(0, 0.015, (img_h, img_w, 3)).astype(np.float32),
                     0, 1)
    img_u8 = (canvas * 255).astype(np.uint8)

    fname = f"bulbs_{idx:04d}.jpg"
    cv2.imwrite(os.path.join(output_dir, fname), img_u8)

    # Pair each centroid with its own ghost list so consumers can read
    # per-bulb shifts straight off the centroid record.  Sort everything by x
    # so positional-rank matching (used by training/inference) lines up.
    bundles = list(zip(xs, ys, radii, cuts, ghosts_per_bulb))
    bundles.sort(key=lambda p: p[0])
    centroids = []
    for x, y, r, cut, bulb_ghosts in bundles:
        entry = {"x": float(x), "y": float(y), "radius": int(r)}
        if cut is None:
            entry["shape"] = "full"
            entry["cut_angle"]  = None
            entry["cut_offset"] = None
        else:
            entry["shape"] = "moon"
            entry["cut_angle"]  = float(cut[0])
            entry["cut_offset"] = float(cut[1])
        entry["ghosts"] = bulb_ghosts
        centroids.append(entry)

    return {
        "image":     fname,
        "img_w":     img_w,
        "img_h":     img_h,
        "intensity": intensity,
        "centroids": centroids,
        "n_ghosts":  n_ghosts,
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
        # Invalidate any stale Stage-1 peak cache from a previous dataset —
        # the new images will have different ghost geometry.
        stale_cache = os.path.join(out_dir, "_blurred_peaks.npy")
        if os.path.isfile(stale_cache):
            os.remove(stale_cache)
            print(f"  removed stale peak cache: {stale_cache}")
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
