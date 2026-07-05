#!/usr/bin/env python3
"""
SUN RGB-D loader for Guided Depth Super-Resolution (GDSR) evaluation.

For every scene it yields an aligned RGB / depth pair prepared the way the GDSR
literature (DKN, FDKN, FDSR, DCTNet, SUFT, ...) sets up synthetic evaluation:

    guide  : RGB image                       (3, H, W)  float in [0, 1]
    lr     : low-res depth (bicubic /s)       (1, H/s, W/s)  meters
    lr_up  : LR bicubically upsampled to HR   (1, H, W)      <- usual model input
    gt     : ground-truth HR depth            (1, H, W)      meters
    mask   : valid-pixel mask (gt > 0)        (1, H, W)      bool

The degradation is: take the (clean) HR depth as ground truth, bicubically
downsample by `scale` to get the LR input, and evaluate the reconstruction with
RMSE. RMSE is reported in centimetres, as is standard for these benchmarks.

Notes
-----
* SUN RGB-D stores depth as a 16-bit PNG whose bits are right-rotated by 3
  (this matches the official toolbox: `bitor(bitshift(d,-3), bitshift(d,13))`).
  We undo that rotation, convert mm -> m, and clamp at `max_depth`.
* `depth_bfx` (cross-bilateral-filtered, hole-filled) is used as GT by default
  because it is denser/cleaner; fall back to `depth` with --depth_type depth.
* A trivial bicubic-upsampling baseline is run in __main__ so you can sanity
  check the pipeline and see where to plug your own model in.

Usage
-----
    python sunrgbd_gdsr.py --root /path/to/SUNRGBD --scale 8 --num 50
"""

import os
import glob
import argparse

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def read_sunrgbd_depth(path, max_depth=8.0):
    """Decode a SUN RGB-D depth PNG into a float32 depth map in metres."""
    d = np.array(Image.open(path)).astype(np.uint16)
    # Undo the 3-bit right rotation used by SUN RGB-D, then mask back to 16 bits.
    d = ((d >> 3) | (d << 13)) & 0xFFFF
    d = d.astype(np.float32) / 1000.0          # millimetres -> metres
    d[d > max_depth] = max_depth               # clamp far/invalid returns
    return d                                    # (H, W)


def find_samples(root, depth_type="depth_bfx"):
    """Walk the SUN RGB-D tree; a sample folder holds both image/ and depth*/ ."""
    samples = []
    for dirpath, dirnames, _ in os.walk(root):
        if "image" in dirnames and depth_type in dirnames:
            imgs = sorted(glob.glob(os.path.join(dirpath, "image", "*.jpg")))
            deps = sorted(glob.glob(os.path.join(dirpath, depth_type, "*.png")))
            if imgs and deps:
                samples.append((imgs[0], deps[0]))
    samples.sort()
    return samples


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SunRGBDGDSR(Dataset):
    def __init__(self, root, scale=8, depth_type="depth_bfx", max_depth=8.0):
        self.samples = find_samples(root, depth_type)
        if not self.samples:
            raise FileNotFoundError(
                f"No 'image/' + '{depth_type}/' pairs found under {root!r}"
            )
        self.scale = scale
        self.max_depth = max_depth

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, dep_path = self.samples[idx]

        depth = read_sunrgbd_depth(dep_path, self.max_depth)          # (H, W)
        H, W = depth.shape

        # RGB is registered to depth but may differ by a few px -> match sizes.
        rgb = Image.open(img_path).convert("RGB").resize((W, H), Image.BICUBIC)
        rgb = np.asarray(rgb, dtype=np.float32) / 255.0               # (H, W, 3)

        # Crop so H, W are divisible by the scale factor.
        s = self.scale
        H, W = (H // s) * s, (W // s) * s
        depth = depth[:H, :W]
        rgb = rgb[:H, :W]

        gt = torch.from_numpy(depth)[None, None]                      # (1,1,H,W)
        guide = torch.from_numpy(rgb).permute(2, 0, 1)[None]          # (1,3,H,W)

        # Synthetic degradation: bicubic downsample -> LR, then bicubic up.
        lr = F.interpolate(gt, size=(H // s, W // s),
                           mode="bicubic", align_corners=False)
        lr_up = F.interpolate(lr, size=(H, W),
                             mode="bicubic", align_corners=False)

        mask = gt > 0

        return {
            "guide": guide.squeeze(0),      # (3, H, W)
            "lr": lr.squeeze(0),            # (1, H/s, W/s)
            "lr_up": lr_up.squeeze(0),      # (1, H, W)
            "gt": gt.squeeze(0),            # (1, H, W)
            "mask": mask.squeeze(0),        # (1, H, W)
            "path": dep_path,
        }


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def rmse_cm(pred, gt, mask):
    """RMSE over valid pixels, in centimetres (inputs in metres)."""
    err = (pred - gt)[mask]
    if err.numel() == 0:
        return float("nan")
    return (err.pow(2).mean().sqrt() * 100.0).item()


# --------------------------------------------------------------------------- #
# Demo / baseline eval
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to SUNRGBD root folder")
    ap.add_argument("--scale", type=int, default=8, choices=[2, 4, 8, 16])
    ap.add_argument("--depth_type", default="depth_bfx",
                    choices=["depth", "depth_bfx"])
    ap.add_argument("--max_depth", type=float, default=8.0)
    ap.add_argument("--num", type=int, default=0,
                    help="Limit number of samples (0 = all)")
    args = ap.parse_args()

    ds = SunRGBDGDSR(args.root, scale=args.scale,
                     depth_type=args.depth_type, max_depth=args.max_depth)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    n = args.num if args.num > 0 else len(ds)
    print(f"Found {len(ds)} samples; evaluating {n} at x{args.scale}")

    total = 0.0
    count = 0
    for i, b in enumerate(loader):
        if i >= n:
            break

        # ---- replace this bicubic baseline with your GDSR model ----
        # pred = model(b["guide"], b["lr_up"])   # -> (B,1,H,W) in metres
        pred = b["lr_up"]
        # ------------------------------------------------------------

        r = rmse_cm(pred, b["gt"], b["mask"])
        if not np.isnan(r):
            total += r
            count += 1
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{n}] running mean RMSE = {total / count:.3f} cm")

    print(f"\nBicubic baseline mean RMSE (x{args.scale}): "
          f"{total / max(count, 1):.3f} cm  over {count} images")


if __name__ == "__main__":
    main()
