"""
prepare_diode.py — convert DIODE train (1024x768) to competition format (640x480).

Run once from the repo root:
    python prepare_diode.py            # converts all images
    python prepare_diode.py --check    # inspects one converted sample

Output layout (one folder per image):
    data/diode_train_480x640/<indoors|outdoor>/<scene>/<scan>/<name>/
        rgb.png     (480, 640, 3) uint8
        depth.npy   (480, 640)    float32, meters
        valid.npy   (480, 640)    bool
    data/diode_train_480x640/intrinsics.json
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

SRC = r"D:\DIODE\train.tar\train"
DST = os.path.join("data", "diode_train_480x640")

OUT_W, OUT_H = 640, 480          # competition resolution (width, height)
SCALE = OUT_W / 1024             # 0.625

# DIODE intrinsics (diode-devkit/intrinsics.txt), valid for all images at 1024x768
FX, FY, CX, CY = 886.81, 927.06, 512.0, 384.0


def convert_one(rgb_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    img = Image.open(rgb_path)
    resized_img = img.resize((OUT_W, OUT_H), resample=Image.Resampling.BOX)
    resized_img.save(os.path.join(out_dir, "rgb.png"))

    depth = np.load(rgb_path.replace('.png', '_depth.npy'))
    depth = np.squeeze(depth, axis=2)

    mask = np.load(rgb_path.replace('.png', '_depth_mask.npy'))
    mask = mask > 0.5
    uint8_mask = mask.astype(np.uint8)

    resized_depth = cv2.resize(depth, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    resized_mask = cv2.resize(uint8_mask, (OUT_W, OUT_H), interpolation=cv2.INTER_NEAREST)
    bool_resized_mask = resized_mask > 0

    valid = (resized_depth > 0) & bool_resized_mask

    np.save(os.path.join(out_dir, "depth.npy"), resized_depth.astype(np.float32))
    np.save(os.path.join(out_dir, "valid.npy"), valid)

def main():
    rgb_paths = sorted(glob.glob(os.path.join(SRC, "**", "*.png"), recursive=True))
    rgb_paths = [p for p in rgb_paths if "indoors" in p]
    print(f"found {len(rgb_paths)} images")          # expect 771
    if not rgb_paths:
        sys.exit("No images found — run from the repo root and check SRC.")

    for i, p in enumerate(rgb_paths, 1):
        rel = os.path.relpath(p, SRC)                  # e.g. indoors\scene_..\scan_..\name.png
        out_dir = os.path.join(DST, os.path.splitext(rel)[0])
        convert_one(p, out_dir)
        if i % 100 == 0 or i == len(rgb_paths):
            print(f"  {i}/{len(rgb_paths)}")

    data = {
        "fx": FX * SCALE,
        "fy": FY * SCALE,
        "cx": CX * SCALE,
        "cy": CY * SCALE,
    }


    with open(os.path.join(DST, "intrinsics.json"), "w") as file:
        json.dump(data, file)
    print("done")


def check():
    """Inspect one converted sample — run after main()."""
    folders = sorted(glob.glob(os.path.join(DST, "**", "rgb.png"), recursive=True))
    print("converted samples:", len(folders),
          "| indoor:", sum("indoors" in f for f in folders),
          "| outdoor:", sum("outdoor" in f for f in folders))

    d = os.path.dirname(folders[0])
    rgb = np.array(Image.open(os.path.join(d, "rgb.png")))
    depth = np.load(os.path.join(d, "depth.npy"))
    valid = np.load(os.path.join(d, "valid.npy"))

    print("sample:", d)
    print("rgb  :", rgb.shape, rgb.dtype)                       # (480, 640, 3) uint8
    print("depth:", depth.shape, depth.dtype)                   # (480, 640) float32
    print("valid:", valid.shape, valid.dtype, f"{valid.mean():.2%} valid")   # (480, 640) bool
    print("valid depth range:", f"{depth[valid].min():.2f} – {depth[valid].max():.2f} m")
    print("intrinsics:", json.load(open(os.path.join(DST, "intrinsics.json"))))

    # visual alignment check: RGB and depth side by side
    import matplotlib.pyplot as plt
    vis = np.where(valid, depth, np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(rgb);                       ax[0].set_title("rgb")
    ax[1].imshow(vis, cmap="Spectral_r");    ax[1].set_title("depth (invalid = blank)")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig("diode_check.png", dpi=100)
    print("saved diode_check.png — edges in both images should line up")


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()


