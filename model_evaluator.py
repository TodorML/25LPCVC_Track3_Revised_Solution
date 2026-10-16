"""
run_eval.py — evaluate ONE model on DIODE val (480x640).

Pick the model below, then run from the repo root:
    python run_eval.py
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
import json
from evaluate import median_scale, abs_rel, delta1, unproject, point_metrics

MODEL = "metric"          # "relative", "relative_inverted" or "metric"
N_PER_SPLIT = 100
DATA = os.path.join("data", "diode_480x640")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join("DepthAnythingV2", "checkpoints")
METRIC_CKPT = os.path.join("runs", "ft_perimg_5ep", "best.pth")
VITS = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
K = json.load(open(os.path.join("data", "diode_480x640", "intrinsics.json")))
#sub = np.zeros((480, 640), dtype=bool)
#sub[::4, ::4] = True                    
# --- 1. load the model --------------------------------------------------------
if MODEL == "metric":
    from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**VITS, max_depth=20)
    model.load_state_dict(torch.load(METRIC_CKPT, map_location="cpu"))
else:
    from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(**VITS)
    model.load_state_dict(torch.load(os.path.join(CKPT, "depth_anything_v2_vits.pth"), map_location="cpu"))
model = model.to(DEVICE).eval()


all_paths = sorted(glob.glob(os.path.join(DATA, "**", "rgb.png"), recursive=True))
indoor = [p for p in all_paths if "indoors" in p]
outdoor = [p for p in all_paths if "outdoor" in p]
splits = {
    "indoors": indoor[::len(indoor)//N_PER_SPLIT],
    "outdoor": outdoor[::len(outdoor)//N_PER_SPLIT],
}
print(f"model: {MODEL}   indoors: {len(splits['indoors'])}   outdoor: {len(splits['outdoor'])}")





# --- 2. loop over all images --------------------------------------------------
results = {"indoors": [], "outdoor": []}
for split, paths in splits.items():
    for rgb_path in paths:
        folder = os.path.dirname(rgb_path)
        rgb = np.array(Image.open(rgb_path).convert("RGB"))     # (480, 640, 3) uint8, 0–255
        gt = np.load(os.path.join(folder, "depth.npy"))          # (480, 640) meters
        valid = np.load(os.path.join(folder, "valid.npy"))       # (480, 640) bool

        # raw 0–255 image in, exactly like on the phone
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(DEVICE)
        with torch.no_grad():
            pred = model(x)[0, 0].cpu().numpy()                  # (480, 640)

        if MODEL == "relative_inverted":
            pred = 1.0 - pred                                 

        with np.errstate(divide="ignore", invalid="ignore"):     # min-max output contains zeros
            pred = median_scale(pred, gt, valid)
            v = valid
            P = unproject(pred, v, K["fx"], K["fy"], K["cx"], K["cy"])
            G = unproject(gt,   v, K["fx"], K["fy"], K["cx"], K["cy"])
            f5  = point_metrics(P, G, tau=0.05)["fscore"]
            f10 = point_metrics(P, G, tau=0.10)["fscore"]

            
            results[split].append((abs_rel(pred, gt, valid), delta1(pred, gt, valid), f5, f10))

# --- 3. print the averages ----------------------------------------------------
print(f"model: {MODEL}")
for split, r in results.items():
    r = np.array(r)
    print(f"  {split:8s} n={len(r):3d}   AbsRel={r[:, 0].mean():.4f}   delta1={r[:, 1].mean():.4f}   "
          f"F@5cm={r[:, 2].mean():.4f}   F@10cm={r[:, 3].mean():.4f}")
r = np.array(results["indoors"] + results["outdoor"])
print(f"  {'all':8s} n={len(r):3d}   AbsRel={r[:, 0].mean():.4f}   delta1={r[:, 1].mean():.4f}, F@5cm={r[:, 2].mean():.4f}, F@10cm={r[:, 3].mean():.4f} ")




