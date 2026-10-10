import numpy as np, torch

def predict(model, rgb_uint8):
    """rgb_uint8: (480, 640, 3) array, values 0–255 → returns (480, 640) prediction."""
    x = torch.from_numpy(rgb_uint8).permute(2, 0, 1)[None].float()   # raw 0–255, like on the phone
    with torch.no_grad():
        return model(x)[0, 0].numpy()

def median_scale(pred, gt, valid):
    # pred, gt: (480, 640) float   valid: (480, 640) bool
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    return pred * np.median(gt_valid) / np.median(pred_valid)

def abs_rel(pred, gt, valid):
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    return np.mean(np.abs(pred_valid - gt_valid) / gt_valid)

def delta1(pred, gt, valid):
    gt_valid = gt[valid]
    pred_valid = pred[valid]
    max_ratios = np.maximum(pred_valid / gt_valid, gt_valid / pred_valid)
    return np.mean(max_ratios < 1.25)

def evaluate(model, samples):
    """samples: list of (rgb, gt_depth, valid_mask), all at 480×640."""
    results = []
    for rgb, gt, valid in samples:
        pred = median_scale(predict(model, rgb), gt, valid)
        results.append((abs_rel(pred, gt, valid), delta1(pred, gt, valid)))
    return np.mean(results, axis=0)
