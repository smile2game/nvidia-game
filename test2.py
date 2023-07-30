import torch
import tensorrt as trt
# time_in = torch.zeros(1, dtype=torch.int64).to("cuda")
# device = time_in.device
# print(time_in)
# H = 256
# W = 384
# control = []
# control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
# control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
# for i in range(13):
#     print(control[i].shape)
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import tensorrt as trt
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

apply_canny = CannyDetector()
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
H = 256
W = 384
trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, '')

with open("./sd_diffusion_fp16.engine", 'rb') as f:
            engine_str = f.read()

control_engine = trt.Runtime(trt_logger).deserialize_cuda_engine(engine_str)
control_context = control_engine.create_execution_context()

# control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
# control_context.set_binding_shape(1, (1, 3, H, W))
# control_context.set_binding_shape(2, (1,))
# control_context.set_binding_shape(3, (1, 77, 768))
