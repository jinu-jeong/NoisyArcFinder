"""
train_label.py
──────────────
Direct (x, y) coordinate-regression trainer for 3-bulb centroid detection.
Companion to train_heatmap.py: same dataset, same resolution, same eval
metric — but the model emits 6 floats (3 × (x, y)) instead of a heatmap.

Pipeline:
  - Load synthetic_data/{train,val,test} + ground_truth.json
  - Resize images to 256×192 (matches train_heatmap.py)
  - Targets are GT centroids sorted by x and normalized to [0, 1]
  - CNN encoder → global-average-pool → FC head → 6 floats, sigmoid-bounded
  - Loss: MSE on normalized coordinates
  - Best checkpoint chosen on val pixel error in original-image pixels
  - Test set is evaluated once at the end using the best checkpoint
  - Best checkpoint → checkpoints/label_best.pt

Split:  train 70 % / val 10 % / test 20 %
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ── Config ─────────────────────────────────────────────────
ORIG_W, ORIG_H = 640, 480
IN_W,   IN_H   = 256, 192      # input resolution (matches train_heatmap.py)
EPOCHS         = 40
BATCH_SIZE     = 32
LR             = 1e-3
N_POINTS       = 3             # 3 bulbs
CKPT           = "checkpoints/label_best.pt"
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── Dataset ────────────────────────────────────────────────
class BulbLabelDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        with open(os.path.join(data_dir, "ground_truth.json")) as f:
            self.records = json.load(f)
        self.data_dir = data_dir

        tfms = [transforms.Resize((IN_H, IN_W))]
        if augment:
            tfms.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        tfms.append(transforms.ToTensor())
        self.transform = transforms.Compose(tfms)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.data_dir, rec["image"])).convert("RGB")
        img = self.transform(img)

        orig_w = rec.get("img_w", ORIG_W)
        orig_h = rec.get("img_h", ORIG_H)

        # GT centroids sorted by x, in original-image px
        sorted_pts = sorted(
            [(c["x"], c["y"]) for c in rec["centroids"]],
            key=lambda p: p[0],
        )
        coords_orig = torch.tensor(sorted_pts, dtype=torch.float32)  # (3, 2)

        # Normalized targets in [0, 1] for the model
        coords_norm = torch.empty_like(coords_orig)
        coords_norm[:, 0] = coords_orig[:, 0] / orig_w
        coords_norm[:, 1] = coords_orig[:, 1] / orig_h

        orig_wh = torch.tensor([orig_w, orig_h], dtype=torch.float32)
        return img, coords_norm, coords_orig, orig_wh


# ── Model ──────────────────────────────────────────────────
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
    )


class CoordRegressor(nn.Module):
    """CNN encoder mirroring train_heatmap's UNet down-path, plus an extra
    pool so the spatial map is 8×6 before global-average pooling.  A small
    FC head produces 2·N_POINTS floats, sigmoid-squashed into [0, 1]."""

    def __init__(self, n_points=N_POINTS):
        super().__init__()
        self.n_points = n_points
        self.d1 = conv_block(3,    16)
        self.d2 = conv_block(16,   32)
        self.d3 = conv_block(32,   64)
        self.d4 = conv_block(64,  128)
        self.d5 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, n_points * 2),
        )

    def forward(self, x):
        x = self.d1(x)
        x = self.d2(self.pool(x))
        x = self.d3(self.pool(x))
        x = self.d4(self.pool(x))
        x = self.d5(self.pool(x))
        x = self.gap(x).flatten(1)        # (B, 256)
        x = self.head(x)                  # (B, 2 · n_points)
        x = torch.sigmoid(x)
        return x.view(-1, self.n_points, 2)   # (B, n_points, 2) in [0, 1]


# ── Helpers ────────────────────────────────────────────────
def to_orig_coords(coords_norm, orig_wh):
    """Scale (B, k, 2) from [0,1] back to each image's original size.
    orig_wh: (B, 2) tensor of [orig_w, orig_h]."""
    out = coords_norm.clone()
    out[..., 0] *= orig_wh[:, 0:1]
    out[..., 1] *= orig_wh[:, 1:2]
    return out


def mean_pixel_error(pred, target):
    return (pred - target).norm(dim=-1).mean().item()


# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)

    train_ds = BulbLabelDataset("synthetic_data/train", augment=True)
    val_ds   = BulbLabelDataset("synthetic_data/val",   augment=False)
    test_ds  = BulbLabelDataset("synthetic_data/test",  augment=False)
    pin = DEVICE == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=pin)

    model     = CoordRegressor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device: {DEVICE}  |  train={len(train_ds)}  val={len(val_ds)}  "
          f"test={len(test_ds)}  |  params={n_params/1e6:.2f}M\n")

    best_err = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for imgs, coords_norm, _, _ in train_loader:
            imgs        = imgs.to(DEVICE)
            coords_norm = coords_norm.to(DEVICE)
            preds       = model(imgs)
            loss        = criterion(preds, coords_norm)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = val_err = 0.0
        with torch.no_grad():
            for imgs, coords_norm, coords_orig, orig_wh in val_loader:
                imgs        = imgs.to(DEVICE)
                coords_norm = coords_norm.to(DEVICE)
                preds       = model(imgs)
                val_loss   += criterion(preds, coords_norm).item() * len(imgs)

                pts_orig    = to_orig_coords(preds.cpu(), orig_wh)
                val_err    += mean_pixel_error(pts_orig, coords_orig) * len(imgs)
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
        for imgs, coords_norm, coords_orig, orig_wh in test_loader:
            imgs        = imgs.to(DEVICE)
            coords_norm = coords_norm.to(DEVICE)
            preds       = model(imgs)
            test_loss  += criterion(preds, coords_norm).item() * len(imgs)

            pts_orig    = to_orig_coords(preds.cpu(), orig_wh)
            test_err   += mean_pixel_error(pts_orig, coords_orig) * len(imgs)
    test_loss /= len(test_ds)
    test_err  /= len(test_ds)
    print(f"Test  — loss={test_loss:.6f}  px_err={test_err:.2f}px")
