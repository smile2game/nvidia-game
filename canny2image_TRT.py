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


class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        H = 256
        W = 384
        # """-----------------------------------------------加载clip的engine模型-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING) #创建logger记录
        trt.init_libnvinfer_plugins(self.trt_logger, '') #初始化插件库
        with open("sd_clip_fp32.plan", 'rb') as f:
            engine_str = f.read() #读取字节1
        clip_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str) #字节序列恢复为对象
        clip_context = clip_engine.create_execution_context() #创建推理的上下文context
        #这里加载进去context
        self.model.cond_stage_model.clip_context = clip_context #替换模型的上下文，与engine是1对多
        print("加载成功clip的engine")
        # """-----------------------------------------------"""

        """-----------------------------------------------加载controlnet的engine模型-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        with open("sd_control_fp16.engine", 'rb') as f:
            engine_str = f.read()
        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        control_context = control_engine.create_execution_context()
        self.model.control_context = control_context
        print("加载成功controlnet的engine")
        """-----------------------------------------------"""

        """-----------------------------------------------加载unet的engine模型-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        with open("sd_diffusion_fp16.engine", 'rb') as f:
            diffusion_engine_str = f.read()
        diffusion_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(diffusion_engine_str)
        diffusion_context = diffusion_engine.create_execution_context()
        self.model.diffusion_context = diffusion_context
        print("加载成功diffusion_model的engine")
        """-----------------------------------------------"""


        """-------------------------提前开buffer----------------------"""
        #controlnet:4 -> 13
        #unet: 3 + 13 -> 1
        #总共：4 +13 +3+1=21
        self.model.control_out = []
        self.model.control_out.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.eps = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to("cuda")
        """-----------------------------------------------"""


    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results