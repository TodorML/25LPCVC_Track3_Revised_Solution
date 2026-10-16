"""
finetune_diode.py — fine-tune the DA-V2 ViT-S metric (Hypersim) model on DIODE train.

DIODE val stays untouched: model selection uses held-out scenes from DIODE train,
and the final comparison is done afterwards with model_evaluator.py.

Before running:
  1. Convert DIODE train with prepare_diode.py
       SRC = data/diode/train   DST = data/diode_train_480x640

Run from the repo root:
    python finetune_diode.py --head-only --save-dir runs/ft_head
"""
import argparse
import glob
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
import cv2
from torch.utils.data import DataLoader, Dataset

from evaluate import median_scale, abs_rel, delta1, unproject, point_metrics
from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

CKPT = os.path.join("DepthAnythingV2", "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth")
VITS = dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384])


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "diode_train_480x640"))
    p.add_argument("--save-dir", default=os.path.join("runs", "ft_head"))
    p.add_argument("--max-depth", type=float, default=20.0)    # must match the checkpoint's head
    p.add_argument("--min-depth", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-7)           # encoder lr; the head gets 10x
    p.add_argument("--head-only", action="store_true")         # freeze the encoder
    p.add_argument("--holdout-every", type=int, default=10)    # every 10th scene → model-selection set
    p.add_argument("--val-images", type=int, default=100)      # ~this many held-out images per validation
    p.add_argument("--low-light", type=float, default=0.0)     # probability of low-light aug
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =============================================================================
# Data
# =============================================================================
def scene_key(rgb_path, root):
    """data/diode_train_480x640/indoors/scene_00001/scan_.../name/rgb.png → 'indoors/scene_00001'"""
    parts = os.path.relpath(rgb_path, root).split(os.sep)
    return parts[0] + "/" + parts[1]


def split_by_scene(root, holdout_every):
    """Hold out whole scenes (not single images): neighbouring scans of one scene are near-duplicates."""
    paths = sorted(glob.glob(os.path.join(root, "**", "rgb.png"), recursive=True))
    scenes = sorted({scene_key(p, root) for p in paths})
    held = set(scenes[::holdout_every])
    train = [p for p in paths if scene_key(p, root) not in held]
    val = [p for p in paths if scene_key(p, root) in held]
    return train, val, held


def strided(paths, n):
    return paths[::max(1, len(paths) // n)]


def low_light(rgb):
    """rgb: (480, 640, 3) float32, 0–255 → darker, phone-ISP-looking version, same range and dtype."""
    lin = (rgb / 255) ** 2.2
    gain = random.uniform(0.05, 0.3)
    sigma = random.uniform(0.3, 1.0)
    darkened = lin * gain

    blurred = cv2.GaussianBlur(darkened, (0, 0), sigma)     # (0, 0) → kernel size computed from sigma
    out = blurred ** (1 / 2.2) * 255
    return np.clip(out, 0, 255).astype(np.float32)


class DiodePrepared(Dataset):
    """Reads the prepare_diode.py output. Returns the image as RAW 0–255 floats, exactly like the phone."""

    def __init__(self, paths, train, low_light_p=0.0):
        self.paths, self.train, self.low_light_p = paths, train, low_light_p

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        folder = os.path.dirname(self.paths[i])
        rgb = np.array(Image.open(self.paths[i]).convert("RGB"), dtype=np.float32)   # (480, 640, 3)
        depth = np.load(os.path.join(folder, "depth.npy")).astype(np.float32)        # (480, 640) m
        valid = np.load(os.path.join(folder, "valid.npy"))                            # (480, 640) bool

        if self.train:
            if random.random() < 0.5:                              # the only aug in the DA-V2 paper
                rgb, depth, valid = rgb[:, ::-1], depth[:, ::-1], valid[:, ::-1]
            if random.random() < self.low_light_p:
                rgb = low_light(rgb)

        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)   # (3, 480, 640)
        return rgb, torch.from_numpy(np.ascontiguousarray(depth)), torch.from_numpy(np.ascontiguousarray(valid))


# =============================================================================
# Model, loss, optimizer
# =============================================================================
def build_model(args, device):
    model = DepthAnythingV2(**VITS, max_depth=args.max_depth)

    checkpoint = torch.load(CKPT, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint, strict=True)

    if args.head_only:
        for name, param in model.named_parameters():
            if 'pretrained' in name:
                param.requires_grad = False

    return model.to(device)


def silog_loss(pred, gt, mask, lambd=1.0, eps=1e-6, min_pixels=10):
    """Per-image scale-invariant log loss, averaged over the batch.
    pred, gt: (B, 480, 640) metres (float32); mask: (B, 480, 640) bool → scalar loss."""
    losses = []
    for i in range(pred.shape[0]):
        m = mask[i]
        if m.sum() < min_pixels:                 # image with almost no valid pixels → skip
            continue
        g = torch.log(pred[i][m] + eps) - torch.log(gt[i][m])
        losses.append(torch.sqrt(torch.mean(g ** 2) - lambd * torch.mean(g) ** 2 + 1e-12))
    return torch.stack(losses).mean()
    


def build_optimizer(model, lr):
    enc = [p for n, p in model.named_parameters() if p.requires_grad and "pretrained" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and "pretrained" not in n]
    groups = [{"params": head, "lr": lr * 10, "base_lr": lr * 10}]
    if enc:
        groups.append({"params": enc, "lr": lr, "base_lr": lr})
    n_train = sum(p.numel() for g in groups for p in g["params"])
    print(f"trainable params: {n_train / 1e6:.2f} M   (encoder {'frozen' if not enc else 'trainable'})")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)



@torch.no_grad()
def validate(model, paths, K, device, max_depth=20.0, min_depth=0.001):
    model.eval()
    rows = []
    for p in paths:
        folder = os.path.dirname(p)
        rgb = np.array(Image.open(p).convert("RGB"))
        gt = np.load(os.path.join(folder, "depth.npy"))
        valid = np.load(os.path.join(folder, "valid.npy"))

        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(device)
        pred = model(x)[0, 0].float().cpu().numpy()

        
        m = valid & (gt >= min_depth) & (gt <= max_depth)      
        g = np.log(pred[m] + 1e-6) - np.log(gt[m])
        silog_img = np.sqrt(max(np.mean(g ** 2) - np.mean(g) ** 2, 0.0))  
        g_mean = np.mean(g)                                     

        
        pred = median_scale(pred, gt, valid)
        P = unproject(pred, valid, K["fx"], K["fy"], K["cx"], K["cy"])
        G = unproject(gt, valid, K["fx"], K["fy"], K["cx"], K["cy"])
        f10 = point_metrics(P, G, tau=0.10)["fscore"]
        rows.append((abs_rel(pred, gt, valid), delta1(pred, gt, valid), f10,
                     silog_img, g_mean, "outdoor" in p))

    r = np.array(rows, dtype=float)
    out = {}
    for name, sel in (("indoors", r[:, 5] == 0), ("outdoor", r[:, 5] == 1), ("all", np.ones(len(r), bool))):
        if sel.any():
            out[name] = {"n": int(sel.sum()),
                         "abs_rel": r[sel, 0].mean(), "delta1": r[sel, 1].mean(), "f10": r[sel, 2].mean(),
                         "silog_img": r[sel, 3].mean(),          # shape error, averaged per image
                         "scale_spread": r[sel, 4].std()}        # how much scale varies image to image
    return out

def selection_score(metrics):
    """Lower = better. metrics is the dict returned by validate()."""
    return -metrics["all"]["f10"]
    
    



def fmt(metrics):
    return "   ".join(f"{k}: AbsRel {v['abs_rel']:.4f} d1 {v['delta1']:.4f} F@10 {v['f10']:.4f} "
                       f"SIimg {v['silog_img']:.4f} spread {v['scale_spread']:.4f}"
                       for k, v in metrics.items())

# =============================================================================
# Main
# =============================================================================
def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda"
    os.makedirs(args.save_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.save_dir, "args.json"), "w"), indent=2)
    K = json.load(open(os.path.join(args.data, "intrinsics.json")))

    train_paths, val_paths, held = split_by_scene(args.data, args.holdout_every)
    val_paths = strided(val_paths, args.val_images)
    print(f"train images: {len(train_paths)}   held-out scenes: {len(held)}   "
          f"val images: {len(val_paths)} (outdoor {sum('outdoor' in p for p in val_paths)})")

    loader = DataLoader(DiodePrepared(train_paths, train=True, low_light_p=args.low_light),
                        batch_size=args.bs, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)

    model = build_model(args, device)
    opt = build_optimizer(model, args.lr)
    scaler = torch.amp.GradScaler("cuda") if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler()

    
    metrics = validate(model, val_paths, K, device)
    best = selection_score(metrics)
    print("epoch 0 (before fine-tuning):", fmt(metrics))
    torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
    log = open(os.path.join(args.save_dir, "log.jsonl"), "a")
    log.write(json.dumps({"epoch": 0, **metrics}) + "\n"); log.flush()

    total_iters, it = args.epochs * len(loader), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for rgb, gt, valid in loader:
            rgb, gt, valid = rgb.to(device, non_blocking=True), gt.to(device), valid.to(device)
            mask = valid & (gt >= args.min_depth) & (gt <= args.max_depth)
            if mask.sum() < 10:
                continue

            
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(rgb)[:, 0]                              # (B, 480, 640)
            if it == 0:
                assert pred.requires_grad, "forward() is not differentiable — check dpt.py for no_grad/detach/numpy"

            loss = silog_loss(pred.float(), gt, mask)               # loss in fp32 (log near 0 in fp16 → nan)
            if not torch.isfinite(loss):
                print(f"  non-finite loss at iter {it}, batch skipped")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            for g in opt.param_groups:                            
                g["lr"] = g["base_lr"] * (1 - it / total_iters) ** 0.9
            it += 1
            running += loss.item()
            if it % 100 == 0:
                print(f"  epoch {epoch} iter {it}/{total_iters}   loss {running / 100:.4f}   "
                      f"lr {opt.param_groups[0]['lr']:.2e}   {time.time() - t0:.0f}s")
                running = 0.0

        metrics = validate(model, val_paths, K, device)
        score = selection_score(metrics)
        print(f"epoch {epoch}:", fmt(metrics))
        log.write(json.dumps({"epoch": epoch, **metrics}) + "\n"); log.flush()

        torch.save(model.state_dict(), os.path.join(args.save_dir, "last.pth"))
        if score < best:
            best = score
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pth"))
            print(f"  new best ({score:.4f}) → best.pth")


if __name__ == "__main__":     # required on Windows when num_workers > 0
    main()




