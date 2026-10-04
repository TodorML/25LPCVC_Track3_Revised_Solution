
import torch
import qai_hub as hub
import requests
import numpy as np
from PIL import Image
import matplotlib 
import cv2 
import os 

from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

width = 640 
height = 480 
encoder = 'vits' 

#########################################################################################################
# Step 0: 
torch_model = DepthAnythingV2(**model_configs[encoder])
torch_model.load_state_dict(torch.load(f'DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
torch_model.eval()

#########################################################################################################
# Step 1: Trace model
input_shape = (1, 3, height, width)
example_input = torch.rand(input_shape)
traced_torch_model = torch.jit.trace(torch_model, example_input)

#########################################################################################################
# Step 2: Compile model
compile_job = hub.submit_compile_job(
    name=f'depth_anything_v2_{encoder}_518x518',
    model=traced_torch_model,
    device = hub.Device("Samsung Galaxy S25"),
    input_specs=dict(image=input_shape),
)
# share the access permission of this compile job with our evaluation server 
compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])


#########################################################################################################
# Step 3: Profile on cloud-hosted device
target_model = compile_job.get_target_model()
profile_job = hub.submit_profile_job(
    name=f'depth_anything_v2_{encoder}_518x518',
    model=target_model,
    device = hub.Device("Samsung Galaxy S25"),
)

#########################################################################################################
# Step 4: Run inference on cloud-hosted device
sample_image_url = ("https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg")
response = requests.get(sample_image_url, stream=True)
response.raw.decode_content = True
image = Image.open(response.raw).resize((width, height))
input_array = np.expand_dims(np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), axis=0).astype(np.float32) # (1, 3, 480, 640)

# Run inference using the on-device model on the input image
inference_job = hub.submit_inference_job(
    name=f'depth_anything_v2_{encoder}_518x518',
    model=target_model,
    device = hub.Device("Samsung Galaxy S25"),
    inputs=dict(image=[input_array]),
)
on_device_output = inference_job.download_output_data()

### vis depth
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
dd = on_device_output['output_0'][0][0]
depth = dd[0]
depth = depth * 255.0
depth = depth.astype(np.uint8)
depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
cv2.imwrite(f'test_tea_{encoder}_518x518.png', depth)
#########################################################################################################
