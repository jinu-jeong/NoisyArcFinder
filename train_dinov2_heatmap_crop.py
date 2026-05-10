"""
train_dinov2_heatmap_crop.py
────────────────────────────
Two-stage 3-bulb centroid detector.

Stage 1 — Locate each bulb cluster (no learning):
  - Downsample the image to a fixed working width
  - Apply a Gaussian blur whose σ is comparable to the inter-bulb-to-ghost
    distance.  Two regimes both work for our purposes:
       • ghosts close to primary → bulb + ghosts merge into one bright blob
       • ghosts far from primary → primary stays the brightest peak; ghost
         peaks are dimmer (alpha ≤ 0.55 / k of primary)
  - Greedy NMS with a suppression radius wider than the maximum
    primary-to-ghost displacement guarantees that any ghost peak near a
    primary is suppressed.  The 3 NMS picks therefore land near the 3
    distinct bulb groups.
  - Map the working-resolution peaks back to the original image.

Stage 2 — Refine each peak (this file's model):
  - Crop a CROP_SIZE × CROP_SIZE patch from the original image, centered on
    the detected peak (zero-padded if the peak is near an edge — rare given
    generate_bulbs.py's 12.5 % margin).
  - Resize the crop to DINO_INPUT and feed it to a DINOv2 ViT-S/14 backbone.
  - A small upsampling decoder produces a 64×64 heatmap whose target is the
    max-composite of Gaussians at the primary *and* each ghost reflection
    visible inside the crop (ghost positions = primary + k·(sx, sy), k≥1,
    using the per-image shared shifts from generate_bulbs.py).
  - Decode greedy NMS + threshold → up to 3 peaks per crop (1 primary + up
    to 2 ghosts).  Primary = the peak closest to the crop center (the
    Stage-1 blurred peak is biased toward the primary because it dominates
    the total intensity).  Sub-pixel via 3×3 weighted centroid; map crop →
    original-image coordinates.

Each image yields 3 training samples (one per detected blob).  Target
association uses x-rank: peaks and GT centroids are both sorted by x and
matched positionally (safe because generate_bulbs.py forces ≥17 % img_w
between bulb groups).

Eval metric: per-image mean L2 pixel error in original-image coords across
all 3 bulbs, matching train_heatmap.py.

Best checkpoint → checkpoints/dinov2_heatmap_crop_best.pt
"""

import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Config ─────────────────────────────────────────────────
CROP_SIZE       = 500     # crop side length in original-image pixels.
                          # Sized so the 2nd ghost is *always* inside the crop:
                          # generate_bulbs.GHOST_MAX_SHIFT_PX = 80 → 2nd ghost is
                          # at ≤160 px per axis from its primary; the Stage-1
                          # blurred peak biases toward the cluster, leaving
                          # ≥250 − 160 = 90 px of margin for ghost radius.
DINO_INPUT      = 224     # DINOv2 ViT-S/14 expects multiples of 14; 16 patches/side
DINO_PATCH      = 14
HEATMAP_OUT     = 64      # decoder output side
HEATMAP_SIGMA   = 2.5     # gaussian σ at HEATMAP_OUT resolution

# Multi-peak per crop: target heatmap is the max-composite of Gaussians at the
# primary GT *and* each ghost position visible inside the crop. At inference
# we run greedy NMS on the predicted heatmap and accept up to 3 peaks
# (primary + up to 2 ghosts), gated by HEATMAP_PEAK_THRESHOLD.
HEATMAP_PEAK_THRESHOLD = 0.3   # peak score below this → treat as "no peak"
HEATMAP_NMS_R          = 4     # NMS suppression radius (heatmap pixels);
                               # < typical ghost-to-primary distance in heatmap
                               # (≥ 10 hm-px even for the smallest images) and
                               # > HEATMAP_SIGMA so a single peak isn't double-picked

# Stage-1 blob detection params (work in a fixed working resolution to keep
# blur/NMS cost independent of input image size)
WORK_W          = 512
BLUR_SIGMA_FRAC = 0.013   # σ as fraction of WORK_W → ~7 px at 512
NMS_R_FRAC      = 0.13    # suppression radius as fraction of WORK_W
                          # > max ghost displacement (≈ 0.078 · img_w in 2-ghost case)
                          # < min inter-bulb distance (= 0.17 · img_w)

EPOCHS          = 30
BATCH_SIZE      = 32
LR              = 5e-4
NUM_WORKERS     = 4

DINO_MODEL      = "facebook/dinov2-small"
FREEZE_BACKBONE = True

CKPT            = "checkpoints/dinov2_heatmap_crop_best.pt"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ImageNet normalization (DINOv2 was trained with these stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Stage 1: blurred-peak detection ────────────────────────
def detect_blurred_peaks(img_bgr, k=3):
    """Return (k, 2) peaks (x, y) in original-image pixels, sorted by x.

    Algorithm:
      1. Downscale to WORK_W width (preserve aspect).
      2. Convert to grayscale, apply Gaussian blur with σ = WORK_W·BLUR_SIGMA_FRAC.
      3. Greedy NMS: take global argmax, zero a disk of radius
         WORK_W·NMS_R_FRAC, repeat k times.
      4. Scale peaks back to original resolution.
    """
    H, W = img_bgr.shape[:2]
    scale = WORK_W / W
    work_h = max(1, int(round(H * scale)))
    small = cv2.resize(img_bgr, (WORK_W, work_h), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    sigma = max(2.0, WORK_W * BLUR_SIGMA_FRAC)
    ksize = max(3, int(sigma * 4) | 1)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    suppress_r = max(1, int(WORK_W * NMS_R_FRAC))
    yg, xg = np.ogrid[:work_h, :WORK_W]
    canvas = blurred.copy()

    peaks_work = []
    for _ in range(k):
        idx = int(canvas.argmax())
        cy, cx = divmod(idx, WORK_W)
        peaks_work.append((cx, cy))
        d2 = (xg - cx) ** 2 + (yg - cy) ** 2
        canvas[d2 <= suppress_r ** 2] = 0.0

    peaks = np.array(peaks_work, dtype=np.float32) / scale
    peaks = peaks[peaks[:, 0].argsort()]   # sort by x to match GT order
    return peaks


# ── Crop helper ────────────────────────────────────────────
def crop_around(img_bgr, peak_xy, size=CROP_SIZE):
    """Zero-padded crop of side `size` centered on integer-rounded `peak_xy`.

    Returns (crop_bgr, top_left_xy).  GT in crop coords = gt - top_left.
    """
    H, W = img_bgr.shape[:2]
    half = size // 2
    px = int(round(float(peak_xy[0])))
    py = int(round(float(peak_xy[1])))
    x0, y0 = px - half, py - half

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + size), min(H, y0 + size)
    pad_l, pad_t = sx0 - x0, sy0 - y0

    crop = np.zeros((size, size, img_bgr.shape[2]), dtype=img_bgr.dtype)
    if sx1 > sx0 and sy1 > sy0:
        crop[pad_t:pad_t + (sy1 - sy0), pad_l:pad_l + (sx1 - sx0)] = \
            img_bgr[sy0:sy1, sx0:sx1]
    return crop, np.array([x0, y0], dtype=np.float32)


# ── Multiprocessing worker for the per-image blurred-peak cache ───
# Defined at module level so Pool can pickle it. Returns (3, 2) float32.
def _peak_worker(arg):
    data_dir, image_name = arg
    img = cv2.imread(os.path.join(data_dir, image_name))
    return detect_blurred_peaks(img)


# ── Dataset ────────────────────────────────────────────────
class BulbCropDataset(Dataset):
    """Each ground-truth image expands to 3 crop samples (one per blurred peak).

    Returned per item:
      crop_t      : (3, DINO_INPUT, DINO_INPUT) normalized for DINOv2
      heatmap_t   : (1, HEATMAP_OUT, HEATMAP_OUT) target gaussian
      crop_origin : (2,) top-left pixel of crop in original image (x, y)
      gt_orig     : (2,) primary GT centroid in original-image pixels
      gt_ghosts   : (2, 2) ghost-1 / ghost-2 GT positions in original-image
                    pixels; rows past n_ghosts are filled with NaN
      n_ghosts    : ()    integer ghost count for this image (0/1/2)
      orig_wh     : (2,) original image (width, height)  — for diagonal norms
      gid         : (2,) (image_idx, peak_rank)
    """

    def __init__(self, data_dir, augment=False):
        with open(os.path.join(data_dir, "ground_truth.json")) as f:
            self.records = json.load(f)
        self.data_dir = data_dir
        self.augment  = augment

        # Flat index: one entry per (image, peak rank).
        self.index = [(i, j) for i in range(len(self.records)) for j in range(3)]

        norm_tfms = [
            transforms.Resize((DINO_INPUT, DINO_INPUT)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        aug_tfms = [
            transforms.Resize((DINO_INPUT, DINO_INPUT)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform = transforms.Compose(aug_tfms if augment else norm_tfms)

        self.yy, self.xx = np.mgrid[:HEATMAP_OUT, :HEATMAP_OUT].astype(np.float32)

        # Cache blurred peaks across epochs — they are deterministic per image.
        # On a fresh dataset the cache build dominates first-run time, so it
        # runs in a multiprocessing pool with a tqdm progress bar; subsequent
        # runs reload the .npy in milliseconds.
        cache = os.path.join(data_dir, "_blurred_peaks.npy")
        if os.path.isfile(cache):
            self.peaks_per_img = np.load(cache)              # (N, 3, 2)
            assert self.peaks_per_img.shape[0] == len(self.records)
        else:
            from multiprocessing import Pool
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm = None

            print(f"  pre-computing blurred peaks for {data_dir} "
                  f"({len(self.records)} images)...")
            args = [(data_dir, rec["image"]) for rec in self.records]
            n_workers = max(1, (os.cpu_count() or 4))
            arr = np.zeros((len(self.records), 3, 2), dtype=np.float32)
            with Pool(processes=n_workers) as pool:
                it = pool.imap(_peak_worker, args, chunksize=8)
                if tqdm is not None:
                    it = tqdm(it, total=len(args), unit="img",
                              desc=f"  {os.path.basename(data_dir)} peaks")
                for i, peaks in enumerate(it):
                    arr[i] = peaks
            np.save(cache, arr)
            self.peaks_per_img = arr
            print(f"  saved peak cache → {cache}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, j = self.index[idx]
        rec  = self.records[i]
        img_path = os.path.join(self.data_dir, rec["image"])
        img_bgr  = cv2.imread(img_path)                       # (H, W, 3) BGR
        H, W     = img_bgr.shape[:2]

        peak = self.peaks_per_img[i, j]                       # (2,) sorted by x

        # GT centroid (full record) sorted by x → positional match to j-th peak.
        # Each centroid carries its own per-bulb ghost shifts, so we keep the
        # whole dict (not just x/y) to read this bulb's ghosts.
        sorted_centroids = sorted(rec["centroids"], key=lambda c: c["x"])
        this_c   = sorted_centroids[j]
        gt_orig  = np.array([this_c["x"], this_c["y"]], dtype=np.float32)
        ghosts_meta = this_c.get("ghosts") or []

        crop_bgr, origin = crop_around(img_bgr, peak, CROP_SIZE)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        crop_t   = self.transform(crop_pil)                   # (3, 224, 224)

        # Multi-peak target heatmap: max-composite over primary + each ghost
        # of *this* bulb-complex whose center lies inside the crop (with a
        # small >3σ outer margin). Ghosts are per-bulb in the new metadata.
        gt_in_crop = gt_orig - origin                          # (2,) primary in crop coords
        peak_pts   = [gt_in_crop]
        for g in ghosts_meta:
            peak_pts.append(
                gt_in_crop + np.array([g["shift_x"], g["shift_y"]],
                                      dtype=np.float32)
            )

        scale_hm = HEATMAP_OUT / CROP_SIZE
        margin   = 3.0 * HEATMAP_SIGMA
        heatmap  = np.zeros((HEATMAP_OUT, HEATMAP_OUT), dtype=np.float32)
        for pt in peak_pts:
            pt_hm = pt * scale_hm
            if (pt_hm[0] < -margin or pt_hm[0] >= HEATMAP_OUT + margin or
                pt_hm[1] < -margin or pt_hm[1] >= HEATMAP_OUT + margin):
                continue   # peak too far outside crop; no useful target signal
            d2 = (self.xx - pt_hm[0]) ** 2 + (self.yy - pt_hm[1]) ** 2
            np.maximum(heatmap, np.exp(-d2 / (2 * HEATMAP_SIGMA ** 2)),
                       out=heatmap)
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)    # (1, H, W)

        # GT ghost positions in original-image coords, padded to (2, 2) with NaN.
        gt_ghosts = np.full((2, 2), np.nan, dtype=np.float32)
        for k, g in enumerate(ghosts_meta[:2]):
            gt_ghosts[k, 0] = float(gt_orig[0]) + float(g["shift_x"])
            gt_ghosts[k, 1] = float(gt_orig[1]) + float(g["shift_y"])
        n_ghosts = int(rec.get("n_ghosts", len(ghosts_meta)))

        return (
            crop_t,
            heatmap_t,
            torch.from_numpy(origin),                         # (2,)
            torch.from_numpy(gt_orig),                        # (2,)
            torch.from_numpy(gt_ghosts),                      # (2, 2) NaN-padded
            torch.tensor(n_ghosts, dtype=torch.long),         # ()
            torch.tensor([W, H], dtype=torch.float32),        # (2,)
            torch.tensor([i, j], dtype=torch.long),           # (image_idx, peak_rank)
        )


# ── Model: DINOv2 backbone + small upsampling decoder ──────
class DinoV2HeatmapCrop(nn.Module):
    def __init__(self, model_name=DINO_MODEL, freeze_backbone=FREEZE_BACKBONE):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        embed_dim   = self.backbone.config.hidden_size                # 384 for vits14
        patch       = self.backbone.config.patch_size                 # 14
        self.tok_h  = DINO_INPUT // patch                             # 16
        self.tok_w  = DINO_INPUT // patch

        # 16×16 → 32×32 → 64×64, with two conv blocks per stage
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x):
        # x: (B, 3, DINO_INPUT, DINO_INPUT) ImageNet-normalized
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(pixel_values=x)
        else:
            out = self.backbone(pixel_values=x)
        # last_hidden_state: (B, 1 + N, C) — drop CLS, keep patch tokens
        tokens = out.last_hidden_state[:, 1:, :]              # (B, N, C)
        B, N, C = tokens.shape
        feats = tokens.transpose(1, 2).reshape(B, C, self.tok_h, self.tok_w)
        return torch.sigmoid(self.decoder(feats))             # (B, 1, 64, 64)


# ── Decoding heatmap → up to 3 centroids ───────────────────
def _subpixel_centroid(plane, cx, cy):
    """3×3 weighted centroid around integer (cx, cy)."""
    H, W = plane.shape
    x0, x1 = max(cx - 1, 0), min(cx + 2, W)
    y0, y1 = max(cy - 1, 0), min(cy + 2, H)
    patch = np.maximum(plane[y0:y1, x0:x1], 0.0)
    s = patch.sum()
    if s < 1e-6:
        return float(cx), float(cy)
    pyy, pxx = np.mgrid[y0:y1, x0:x1]
    return (
        float((patch * pxx).sum() / s),
        float((patch * pyy).sum() / s),
    )


def decode_crop_multi_peak(
    hm,
    k_max=3,
    suppress_r=HEATMAP_NMS_R,
    threshold=HEATMAP_PEAK_THRESHOLD,
):
    """hm: (B, 1, H, W).

    Returns a list of length B, each a (P, 3) numpy array with rows
    [x, y, score] at heatmap resolution; P ∈ [0, k_max] depending on how many
    peaks survive the threshold.  Greedy NMS, then 3×3 sub-pixel refinement.
    """
    B, _, H, W = hm.shape
    hm_np = hm.detach().cpu().numpy()
    yg, xg = np.ogrid[:H, :W]

    out = []
    for b in range(B):
        plane = hm_np[b, 0]
        canvas = plane.copy()
        peaks = []
        for _ in range(k_max):
            idx = int(canvas.argmax())
            cy, cx = divmod(idx, W)
            score = float(canvas[cy, cx])
            if score < threshold:
                break
            sx, sy = _subpixel_centroid(plane, cx, cy)
            peaks.append([sx, sy, score])
            d2 = (xg - cx) ** 2 + (yg - cy) ** 2
            canvas[d2 <= suppress_r ** 2] = 0.0
        out.append(np.array(peaks, dtype=np.float32).reshape(-1, 3))
    return out


def primary_from_peaks(peaks_per_crop, fallback_xy=(HEATMAP_OUT / 2.0,
                                                    HEATMAP_OUT / 2.0)):
    """For each crop, pick the peak closest to the crop center as the primary.

    Stage-1 places the crop center near the bulb's primary (the brightest
    contributor in the blurred image), so closeness-to-center is a robust
    primary identifier even though target Gaussians are equal-amplitude.

    peaks_per_crop : list[B] of (P, 3) arrays from decode_crop_multi_peak.
    Returns        : (B, 2) tensor of primary (x, y) at heatmap resolution.
                     Crops with no peaks above threshold fall back to the
                     crop center (Stage-1 peak).
    """
    B = len(peaks_per_crop)
    out = torch.zeros(B, 2)
    cx_c = HEATMAP_OUT / 2.0
    cy_c = HEATMAP_OUT / 2.0
    for b, pks in enumerate(peaks_per_crop):
        if pks.shape[0] == 0:
            out[b, 0], out[b, 1] = fallback_xy
            continue
        d2 = (pks[:, 0] - cx_c) ** 2 + (pks[:, 1] - cy_c) ** 2
        i = int(d2.argmin())
        out[b, 0] = float(pks[i, 0])
        out[b, 1] = float(pks[i, 1])
    return out


def crop_hm_to_orig(coords_hm, crop_origin):
    """coords_hm: (B, 2) in heatmap pixels.  crop_origin: (B, 2) top-left of
    the crop in original image pixels.  Returns coords in original-image px.
    """
    coords_crop = coords_hm * (CROP_SIZE / HEATMAP_OUT)        # (B, 2)
    return coords_crop + crop_origin


def crop_errors(peaks_hm, origin_xy, gt_primary_xy, gt_ghosts_xy, n_ghosts):
    """Per-crop primary / ghost-1 / ghost-2 L2 errors in original-image pixels.

    peaks_hm        : (P, 3) decoded peaks at heatmap resolution [x, y, score].
    origin_xy       : (2,)   crop top-left in original-image coords.
    gt_primary_xy   : (2,)   GT primary in original-image coords.
    gt_ghosts_xy    : (2, 2) GT ghost-1 / ghost-2 in original coords; NaN if absent.
    n_ghosts        : int    image-level ghost count (0/1/2).

    Returns (prim_err, g1_err_or_None, g2_err_or_None).  A ghost error is
    reported only when both the GT ghost and a corresponding predicted ghost
    exist for this crop.  Predicted ghosts are ordered by distance from the
    predicted primary, which mirrors the k=1 (closer) / k=2 (further) GT order.
    """
    scale_crop = CROP_SIZE / HEATMAP_OUT
    cx_c, cy_c = HEATMAP_OUT / 2.0, HEATMAP_OUT / 2.0

    if peaks_hm.shape[0] == 0:
        # Total miss — fall back to crop center for primary, no ghosts predicted.
        pred_p_hm = np.array([cx_c, cy_c], dtype=np.float32)
        ghosts_hm = np.zeros((0, 2), dtype=np.float32)
    else:
        d2c = (peaks_hm[:, 0] - cx_c) ** 2 + (peaks_hm[:, 1] - cy_c) ** 2
        pi  = int(d2c.argmin())
        pred_p_hm = peaks_hm[pi, :2]
        ghosts_hm = np.delete(peaks_hm[:, :2], pi, axis=0)
        if ghosts_hm.shape[0] > 1:
            d2g = ((ghosts_hm - pred_p_hm) ** 2).sum(axis=1)
            ghosts_hm = ghosts_hm[np.argsort(d2g)]

    pred_p_orig = pred_p_hm * scale_crop + origin_xy
    prim_err = float(np.linalg.norm(pred_p_orig - gt_primary_xy))

    g1_err = None
    if (n_ghosts >= 1 and ghosts_hm.shape[0] >= 1
            and not np.isnan(gt_ghosts_xy[0]).any()):
        pg1 = ghosts_hm[0] * scale_crop + origin_xy
        g1_err = float(np.linalg.norm(pg1 - gt_ghosts_xy[0]))

    g2_err = None
    if (n_ghosts >= 2 and ghosts_hm.shape[0] >= 2
            and not np.isnan(gt_ghosts_xy[1]).any()):
        pg2 = ghosts_hm[1] * scale_crop + origin_xy
        g2_err = float(np.linalg.norm(pg2 - gt_ghosts_xy[1]))

    return prim_err, g1_err, g2_err


def evaluate(model, loader, criterion):
    """Run the model over a loader and return loss + (prim, g1, g2, counts)."""
    model.eval()
    total_loss = 0.0
    n_seen     = 0
    per_img_prim = {}                  # img_idx → list of per-crop prim errs
    g1_errs, g2_errs = [], []          # crop-level error pools
    with torch.no_grad():
        for (crops, hm_target, origin, gt_prim, gt_ghosts, n_ghosts_b,
             _wh, gid) in loader:
            crops     = crops.to(DEVICE, non_blocking=True)
            hm_target = hm_target.to(DEVICE, non_blocking=True)
            preds     = model(crops)
            total_loss += criterion(preds, hm_target).item() * len(crops)
            n_seen     += len(crops)

            peaks_per_crop = decode_crop_multi_peak(preds)
            origin_np   = origin.numpy()
            gt_prim_np  = gt_prim.numpy()
            gt_ghosts_np = gt_ghosts.numpy()
            for b in range(len(crops)):
                pe, g1, g2 = crop_errors(
                    peaks_per_crop[b], origin_np[b],
                    gt_prim_np[b], gt_ghosts_np[b], int(n_ghosts_b[b]),
                )
                img_i = int(gid[b, 0])
                per_img_prim.setdefault(img_i, []).append(pe)
                if g1 is not None: g1_errs.append(g1)
                if g2 is not None: g2_errs.append(g2)

    avg_loss  = total_loss / max(1, n_seen)
    prim_err  = (sum(sum(v) / len(v) for v in per_img_prim.values())
                 / max(1, len(per_img_prim)))
    g1_mean   = (sum(g1_errs) / len(g1_errs)) if g1_errs else float("nan")
    g2_mean   = (sum(g2_errs) / len(g2_errs)) if g2_errs else float("nan")
    return avg_loss, prim_err, g1_mean, g2_mean, len(g1_errs), len(g2_errs)


def per_image_pixel_error(pred_orig, gt_orig, group_id):
    """Average L2 distance per image (3 crops per image).

    pred_orig, gt_orig : (B, 2) in original-image pixels.
    group_id           : (B, 2) of (image_idx, peak_rank).  Same image_idx → group.
    Returns mean error in pixels averaged over images, plus per-sample errors.
    """
    err = (pred_orig - gt_orig).norm(dim=-1)                  # (B,)
    image_idx = group_id[:, 0].tolist()
    bucket = {}
    for i, e in zip(image_idx, err.tolist()):
        bucket.setdefault(i, []).append(e)
    img_means = [sum(v) / len(v) for v in bucket.values()]
    return sum(img_means) / max(1, len(img_means)), err


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

    print("Building datasets (will pre-compute & cache blurred peaks)...")
    train_ds = BulbCropDataset("synthetic_data/train", augment=True)
    val_ds   = BulbCropDataset("synthetic_data/val",   augment=False)
    test_ds  = BulbCropDataset("synthetic_data/test",  augment=False)
    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin)

    model = DinoV2HeatmapCrop().to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    if os.path.isfile(CKPT):
        try:
            model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
            print(f"Resumed from checkpoint: {CKPT}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}), starting from scratch.")
    else:
        print("No checkpoint found, starting from scratch.")

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in trainable)
    print(f"Device: {DEVICE}  |  "
          f"train_imgs={len(train_ds.records)}  val_imgs={len(val_ds.records)}  "
          f"test_imgs={len(test_ds.records)}  |  "
          f"crops/epoch={len(train_ds)}  |  "
          f"params total={n_total/1e6:.2f}M trainable={n_train/1e6:.2f}M\n")

    best_err = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            crops     = batch[0].to(DEVICE, non_blocking=True)
            hm_target = batch[1].to(DEVICE, non_blocking=True)
            preds     = model(crops)
            loss      = criterion(preds, hm_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(crops)
            n_seen     += len(crops)
        train_loss /= max(1, n_seen)

        # Validation — primary, ghost-1, ghost-2 errors all reported.
        # Best checkpoint selection still uses primary (the one always defined).
        val_loss, val_prim, val_g1, val_g2, n_g1, n_g2 = evaluate(
            model, val_loader, criterion
        )

        scheduler.step()
        marker = ""
        if val_prim < best_err:
            best_err = val_prim
            torch.save(model.state_dict(), CKPT)
            marker = "  ← best"

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"val_px_err  prim={val_prim:5.2f}  "
              f"g1={val_g1:5.2f}(n={n_g1})  "
              f"g2={val_g2:5.2f}(n={n_g2}){marker}")

    # ── Test eval with best checkpoint ─────────────────────
    print(f"\nBest val primary err: {best_err:.2f}px  →  loading {CKPT} for test eval")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    test_loss, test_prim, test_g1, test_g2, t_n_g1, t_n_g2 = evaluate(
        model, test_loader, criterion
    )
    print(f"Test  — loss={test_loss:.6f}  "
          f"px_err  prim={test_prim:.2f}  "
          f"g1={test_g1:.2f}(n={t_n_g1})  "
          f"g2={test_g2:.2f}(n={t_n_g2})")
