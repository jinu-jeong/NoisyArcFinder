"""
train_heatmap.py
────────────────
Single-file heatmap-regression trainer for 3-bulb centroid detection.

Pipeline:
  - Load synthetic_data/{train,val,test} + ground_truth.json
  - Resize images to 256×192 (preserves 4:3 aspect of original 640×480)
  - Build a 1-channel target heatmap with 3 Gaussian peaks at GT primaries
    (ghosts are deliberately NOT in the target — model learns to ignore them)
  - Train a small UNet to regress the heatmap (MSE loss, sigmoid output)
  - Decode with GREEDY DISTANCE-BASED NMS:
      find global max → suppress disk of radius SUPPRESS_R → repeat ×3
      + 3×3 sub-pixel weighted centroid → sort by x to match GT order
    This avoids the max-pool-equality NMS plateau bug.
  - Best checkpoint saved based on val set pixel error (val is NOT test).
  - Test set is evaluated only once at the end using the best checkpoint.
  - Save best checkpoint to checkpoints/heatmap_best.pt

Split:  train 70 % / val 10 % / test 20 %  (2000 images total)
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Config ─────────────────────────────────────────────────
ORIG_W, ORIG_H = 640, 480
IN_W,   IN_H   = 256, 192      # input/output resolution (4:3, divisible by 16)
HEATMAP_SIGMA  = 3.0            # gaussian peak σ in output-resolution pixels
SUPPRESS_R     = 20             # greedy NMS suppression radius (output pixels)
                                # safely > min primary-to-primary dist (≈44 px)
                                # and > typical primary-to-ghost dist after training
EPOCHS         = 40
BATCH_SIZE     = 32
LR             = 1e-3
CKPT           = "checkpoints/heatmap_best.pt"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── Dataset ────────────────────────────────────────────────
class BulbHeatmapDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        with open(os.path.join(data_dir, "ground_truth.json")) as f:
            self.records = json.load(f)
        self.data_dir = data_dir

        tfms = [transforms.Resize((IN_H, IN_W))]
        if augment:
            tfms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        tfms.append(transforms.ToTensor())
        self.transform = transforms.Compose(tfms)

        self.yy, self.xx = np.mgrid[:IN_H, :IN_W].astype(np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.data_dir, rec["image"])).convert("RGB")
        img = self.transform(img)

        # 1-channel heatmap: max-composite of all 3 primary Gaussian peaks
        # Ghosts are NOT in the target — the model must learn to ignore them
        # Use actual image dimensions (may vary per image)
        orig_w = rec.get("img_w", ORIG_W)
        orig_h = rec.get("img_h", ORIG_H)
        sx = IN_W / orig_w
        sy = IN_H / orig_h

        heatmap = np.zeros((IN_H, IN_W), dtype=np.float32)
        sorted_pts = sorted(
            [(c["x"], c["y"]) for c in rec["centroids"]],
            key=lambda p: p[0],
        )
        for gx, gy in sorted_pts:
            cx = gx * sx
            cy = gy * sy
            d2 = (self.xx - cx) ** 2 + (self.yy - cy) ** 2
            heatmap = np.maximum(heatmap, np.exp(-d2 / (2 * HEATMAP_SIGMA ** 2)))
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)

        coords = torch.tensor(sorted_pts, dtype=torch.float32)   # (3, 2) in original px
        orig_wh = torch.tensor([orig_w, orig_h], dtype=torch.float32)  # for decode scaling
        return img, heatmap_t, coords, orig_wh


# ── UNet ───────────────────────────────────────────────────
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = conv_block(3,   16)
        self.d2 = conv_block(16,  32)
        self.d3 = conv_block(32,  64)
        self.bn = conv_block(64, 128)
        self.u3 = conv_block(128 + 64, 64)
        self.u2 = conv_block( 64 + 32, 32)
        self.u1 = conv_block( 32 + 16, 16)
        self.out  = nn.Conv2d(16, 1, 1)
        self.pool = nn.MaxPool2d(2)

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        bn = self.bn(self.pool(d3))
        u3 = self.u3(torch.cat([self._up(bn), d3], dim=1))
        u2 = self.u2(torch.cat([self._up(u3), d2], dim=1))
        u1 = self.u1(torch.cat([self._up(u2), d1], dim=1))
        return torch.sigmoid(self.out(u1))   # (B, 1, H, W) in [0, 1]


# ── Decoding: greedy distance-based NMS + sub-pixel centroid
def decode_heatmap(hm, k=3, suppress_r=SUPPRESS_R):
    """hm: (B, 1, H, W). Returns (B, k, 2) centroids at output resolution (x, y).

    Algorithm per image:
      1. Find global argmax → this is peak 1.
      2. Zero out a disk of radius suppress_r around it.
      3. Repeat to find peaks 2 and 3.
      4. Sub-pixel refinement: 3×3 weighted centroid around each peak.

    Works even with saturated plateaus (takes global max regardless of ties).
    Works even if ghost residual peaks exist (suppression radius larger than
    primary-to-ghost distance → ghost near a found primary is suppressed).
    """
    B, _, H, W = hm.shape
    hm_np = hm.detach().cpu().numpy()
    refined = torch.zeros(B, k, 2)

    # Pre-build a circular suppression mask template
    ys_grid, xs_grid = np.ogrid[:H, :W]

    for b in range(B):
        canvas = hm_np[b, 0].copy()
        peaks = []

        for _ in range(k):
            idx = int(canvas.argmax())
            cy, cx = divmod(idx, W)
            peaks.append((cx, cy))
            # Suppress disk
            dist2 = (xs_grid - cx) ** 2 + (ys_grid - cy) ** 2
            canvas[dist2 <= suppress_r ** 2] = 0.0

        for i, (cx, cy) in enumerate(peaks):
            x0, x1 = max(cx - 1, 0), min(cx + 2, W)
            y0, y1 = max(cy - 1, 0), min(cy + 2, H)
            patch = np.maximum(hm_np[b, 0, y0:y1, x0:x1], 0.0)
            s = patch.sum()
            if s < 1e-6:
                refined[b, i, 0] = float(cx)
                refined[b, i, 1] = float(cy)
            else:
                pyy, pxx = np.mgrid[y0:y1, x0:x1]
                refined[b, i, 0] = float((patch * pxx).sum() / s)
                refined[b, i, 1] = float((patch * pyy).sum() / s)

    return refined  # (B, k, 2) at output resolution


def to_orig_coords(coords_out, orig_wh):
    """Scale (B, k, 2) from output resolution back to each image's original size.
    orig_wh: (B, 2) tensor of [orig_w, orig_h] per sample.
    """
    out = coords_out.clone()
    out[..., 0] *= orig_wh[:, 0:1] / IN_W   # x  (B,1) broadcast over k
    out[..., 1] *= orig_wh[:, 1:2] / IN_H   # y
    return out


def sort_by_x(coords):
    """coords: (B, k, 2). Sort each sample by x so it matches GT order."""
    result = torch.zeros_like(coords)
    for b in range(coords.size(0)):
        result[b] = coords[b, coords[b, :, 0].argsort()]
    return result


def mean_pixel_error(pred, target):
    return (pred - target).norm(dim=-1).mean().item()


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

    train_ds = BulbHeatmapDataset("synthetic_data/train", augment=True)
    val_ds   = BulbHeatmapDataset("synthetic_data/val",   augment=False)
    test_ds  = BulbHeatmapDataset("synthetic_data/test",  augment=False)
    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)

    model     = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # Load existing checkpoint as initial weights if available
    if os.path.isfile(CKPT):
        try:
            model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
            print(f"Resumed from checkpoint: {CKPT}")
        except Exception as e:
            print(f"Could not load checkpoint ({e}), starting from scratch.")
    else:
        print("No checkpoint found, starting from scratch.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}  |  train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  |  params={n_params/1e6:.2f}M\n")

    best_err = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, hm_target, _, __ in train_loader:
            imgs      = imgs.to(DEVICE)
            hm_target = hm_target.to(DEVICE)
            preds     = model(imgs)
            loss      = criterion(preds, hm_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = val_err = 0.0
        with torch.no_grad():
            for imgs, hm_target, coords_orig, orig_wh in val_loader:
                imgs      = imgs.to(DEVICE)
                hm_target = hm_target.to(DEVICE)
                preds     = model(imgs)
                val_loss += criterion(preds, hm_target).item() * len(imgs)

                pts_out  = decode_heatmap(preds)
                pts_orig = sort_by_x(to_orig_coords(pts_out, orig_wh))
                val_err += mean_pixel_error(pts_orig, coords_orig) * len(imgs)
        val_loss /= len(val_ds)
        val_err  /= len(val_ds)

        scheduler.step()
        marker = ""
        if val_err < best_err:
            best_err = val_err
            torch.save(model.state_dict(), CKPT)
            marker = "  ← best"

        print(f"Epoch {epoch:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"val_px_err={val_err:.2f}px{marker}")

    # ── Final evaluation on held-out test set (best checkpoint) ──
    print(f"\nBest val pixel error: {best_err:.2f}px  →  loading {CKPT} for test eval")
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()
    test_loss = test_err = 0.0
    with torch.no_grad():
        for imgs, hm_target, coords_orig, orig_wh in test_loader:
            imgs      = imgs.to(DEVICE)
            hm_target = hm_target.to(DEVICE)
            preds     = model(imgs)
            test_loss += criterion(preds, hm_target).item() * len(imgs)

            pts_out  = decode_heatmap(preds)
            pts_orig = sort_by_x(to_orig_coords(pts_out, orig_wh))
            test_err += mean_pixel_error(pts_orig, coords_orig) * len(imgs)
    test_loss /= len(test_ds)
    test_err  /= len(test_ds)
    print(f"Test  — loss={test_loss:.6f}  px_err={test_err:.2f}px")
