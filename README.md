# LPCVC 2025 Track 3 — Monocular Depth Estimation

Forked from [lpcvai/25LPCVC_Track3_Sample_Solution](https://github.com/lpcvai/25LPCVC_Track3_Sample_Solution).

This repo documents my work on Track 3 of the 2025 Low Power Computer Vision Challenge: optimizing and fine-tuning [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-S) for metric depth estimation on a Samsung Galaxy S25, evaluated via Qualcomm AI Hub. The optimization part was done by IEEE LPCVC 2025 winner solution - Sailor Moon. 
Please refer to the repo: https://github.com/chen7545/LPCVC2025_SailorMoon_Track3 for that part.

---

## What I did

### Reproducing the winning solution (Runs A–D)

The original sample solution runs at 26 ms (518×518 input, relative model). I reproduced four configurations to understand the latency-accuracy trade-off:

| Run | Model | Input resolution | Inference time (Galaxy S25) |
|-----|-------|------------------|-----------------------------|
| A | Relative (ViT-S) | 518×518 | 26 ms |
| B | Relative (ViT-S) | 322×434 | 14.6 ms |
| C | Metric Hypersim (ViT-S) | 322×434 | **14.2 ms** |
| D | Metric Hypersim, fused norm (ViT-S) | 322×434 | **14.2 ms** |

**Key insight:** the 1.78× speedup (26 ms → 14.2 ms) comes entirely from reducing the input resolution from 518×518 to 322×434. Relative and metric are the same ViT-S architecture. Switching to the metric checkpoint additionally gives absolute depth output in meters, which is required for the 3D evaluation metrics.

**Run D** fuses the ImageNet normalization into the patch embedding weights at export time (`fused_norm=True`), removing the mean/std subtraction from the runtime graph.

To make the input resolution configurable for the relative model (runs A/B), I added `self.input_size` to the relative `dpt.py` so it can be set before tracing:
```python
torch_model.input_size = (322, 434)   # set before torch.jit.trace
```
The metric `dpt.py` (runs C/D) hardcodes `(322, 434)` directly in `forward()`.

---

### Fine-tuning on DIODE

The metric model was pretrained on Hypersim (synthetic indoor data). I fine-tuned it on real sensor data from the [DIODE](https://diode-dataset.org/) indoor split to improve accuracy on the evaluation benchmark for DIODE.

**Data preparation** (`prepare_diode.py`): converts DIODE train images from 1024×768 to the competition resolution (640×480) and rescales camera intrinsics accordingly.

**Training** (`fine_tune.py`):
- Loss: scale-invariant log loss (SILog)
- Split: scene-level holdout to prevent data leakage from near-duplicate scan viewpoints
- Encoder frozen (`--head-only`), head trained at lr = 5e-6 with AdamW
- Schedule: polynomial decay over total iterations (same as original DA-V2 training)
- Model selection: based on 3D F-score at 10 cm threshold on the held-out scenes

**Evaluation** (`model_evaluator.py`, `evaluate.py`): reports AbsRel, δ1, F@5cm, and F@10cm. The 3D metrics unproject both predicted and ground-truth depth maps to point clouds and compute precision/recall/F-score using nearest-neighbour search.

---

## Results on DIODE val

F@10cm evaluated on a strided subset of DIODE val (109 indoor, 112 outdoor images). Fine-tuning used indoor train data only, so the outdoor improvement is out-of-distribution generalization.

| Model | Indoor F@10cm ↑ | Outdoor F@10cm ↑ |
|-------|----------------|-----------------|
| Metric Hypersim baseline | 59.95 | 15.43 |
| + DIODE head fine-tune | 63.57 | 18.22 |



---

## Repository structure

```
demo_run_A.py          # Run A: relative model, 518×518
demo_run_B.py          # Run B: relative model, 322×434
demo_run_C.py          # Run C: metric model, 322×434
demo_run_D.py          # Run D: metric model, fused norm, 322×434 - Sailor moon's winning solution
prepare_diode.py       # Convert DIODE train to 640×480 competition format
fine_tune.py           # Fine-tuning pipeline
model_evaluator.py     # Evaluation script (AbsRel, δ1, F@5cm, F@10cm)
evaluate.py            # Metric functions and 3D point cloud utilities
DepthAnythingV2/       # DA-V2 source (modified dpt.py files)
out_vis/               # Sample inference outputs
```

---

## Setup

```bash
pip install torch torchvision opencv-python pillow scipy qai-hub
```

Configure Qualcomm AI Hub:
```bash
pip install qai-hub
qai-hub configure --api_token YOUR_TOKEN
```

Fine-tune:
```bash
# 1. Convert DIODE train dataset
python prepare_diode.py

# 2. Fine-tune (head only, 5 epochs)
python fine_tune.py --head-only --epochs 5 --save-dir runs/ft_head

# 3. Evaluate on DIODE val
python model_evaluator.py
```

Run on device:
```bash
python demo_run_C.py   # metric model, 322×434, 14.2 ms on Galaxy S25
python demo_run_D.py   # same with fused normalization
```

---

## Sample outputs

| Run B — Relative 322×434 | Run D — Metric fused 322×434 |
|:---:|:---:|
| ![B](out_vis/B_relative_322x434.png) | ![D](out_vis/D_metric_hypersim_fused_322x434.png) |
