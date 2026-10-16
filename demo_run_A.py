import torch
import qai_hub as hub
import requests
import numpy as np
from PIL import Image
import matplotlib
import cv2
import os

from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2   # RELATIVE model

RUN        = "A_relative_518x518"
INPUT_SIZE = (518, 518)                       # (H, W)
DEVICE     = hub.Device("Samsung Galaxy S25")

width, height = 640, 480

# Step 0: model
torch_model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
torch_model.load_state_dict(torch.load(
    'DepthAnythingV2/checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
torch_model.input_size = INPUT_SIZE            # must be set BEFORE tracing
torch_model.eval()

# Step 1: trace
input_shape = (1, 3, height, width)
traced_torch_model = torch.jit.trace(torch_model, torch.rand(input_shape) * 255)

# Step 2: compile
compile_job = hub.submit_compile_job(
    name=RUN, model=traced_torch_model, device=DEVICE,
    input_specs=dict(image=input_shape),
)
target_model = compile_job.get_target_model()

# Step 3: inference on the sample image, with profiling
url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
response = requests.get(url, stream=True)
response.raw.decode_content = True
image = Image.open(response.raw).resize((width, height))
input_array = np.expand_dims(np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), 0)

inference_job = hub.submit_inference_job(
    name=RUN, model=target_model, device=DEVICE,
    inputs=dict(image=[input_array]),
    profile=True,
)
out = inference_job.download_output_data()['output_0'][0]   # (1, 1, 480, 640)
depth = out[0, 0]
print(f"output range: {depth.min():.3f} – {depth.max():.3f}  (relative model: expect 0.000 – 1.000)")

# Visualization
vis = (depth - depth.min()) / (depth.max() - depth.min())
vis = (matplotlib.colormaps.get_cmap('Spectral_r')(
    (vis * 255).astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
os.makedirs('out_vis', exist_ok=True)
cv2.imwrite(f'out_vis/{RUN}.png', vis)