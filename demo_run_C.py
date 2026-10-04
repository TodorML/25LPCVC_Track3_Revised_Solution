import torch
import qai_hub as hub
import requests
import numpy as np
from PIL import Image
import matplotlib
import cv2
import os

from DepthAnythingV2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2   # METRIC

RUN    = "C_metric_hypersim_322x434"
DEVICE = hub.Device("Samsung Galaxy S25")

width, height = 640, 480
encoder = 'vits'

# Step 0: model
torch_model = DepthAnythingV2(encoder='vits', features=64,
                              out_channels=[48, 96, 192, 384], max_depth=20)
torch_model.load_state_dict(torch.load(
    'DepthAnythingV2/checkpoints/depth_anything_v2_metric_hypersim_vits.pth',
    map_location='cpu'))
torch_model.eval()

# Step 1: trace
input_shape = (1, 3, height, width)
traced_torch_model = torch.jit.trace(torch_model, torch.rand(input_shape) * 255)

# Step 2: compile
compile_job = hub.submit_compile_job(
    name=RUN, model=traced_torch_model, device=DEVICE,
    input_specs=dict(image=input_shape),
)

# Step 3: profile
target_model = compile_job.get_target_model()
# profile_job = hub.submit_profile_job(name=RUN, model=target_model, device=DEVICE)

# Step 4: inference on the sample image
url = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
response = requests.get(url, stream=True)
response.raw.decode_content = True
image = Image.open(response.raw).resize((width, height))
input_array = np.expand_dims(np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), 0)

inference_job = hub.submit_inference_job(
    name=RUN, model=target_model, device=DEVICE, inputs=dict(image=[input_array]), profile = True,
)
out = inference_job.download_output_data()['output_0'][0]   # (1, 1, 480, 640), meters
depth = out[0, 0]
print(f"depth range: {depth.min():.2f} – {depth.max():.2f} m")

# Visualization: min-max ONLY for display (output is meters, not [0,1])
vis = (depth - depth.min()) / (depth.max() - depth.min())
vis = (matplotlib.colormaps.get_cmap('Spectral_r')(
    (vis * 255).astype(np.uint8))[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
os.makedirs('out_vis', exist_ok=True)
cv2.imwrite(f'out_vis/{RUN}.png', vis)