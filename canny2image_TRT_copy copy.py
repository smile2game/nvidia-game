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
        """----------------------------------------------转换cond_stage_model为engine-----------------"""
        cond_stage_model = self.model.cond_stage_model
        """-----------------------------------------------"""




        """-----------------------------------------------转换control_model为engine-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        control_model = self.model.control_model
        if not os.path.isfile("sd_control_fp16.engine"):
            x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
            h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
            t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
            c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
            controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)
            output_names = []
            for i in range(13): #这里还不知道为什么是13
                output_names.append("out_"+ str(i))
            dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                                'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                                't_in' : {0 : 'bs'}, 
                                'c_in' : {0 : 'bs'}} #这里是需要看一下怎么写的
            for i in range(13):
                dynamic_table[output_names[i]] = {0 : "bs"}
            torch.onnx.export(control_model,               
                                (x_in, h_in, t_in, c_in),  
                                "./sd_control_test.onnx",   
                                export_params=True,
                                opset_version=16,
                                do_constant_folding=True,
                                keep_initializers_as_inputs=True,
                                input_names = ['x_in', "h_in", "t_in", "c_in"], 
                                output_names = output_names, 
                                dynamic_axes = dynamic_table)
            os.system("trtexec --onnx=sd_control_test.onnx --saveEngine=sd_control_fp16.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")
        
        with open("./sd_control_fp16.engine", 'rb') as f:
            engine_str = f.read()
        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        control_context = control_engine.create_execution_context()
        control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
        control_context.set_binding_shape(1, (1, 3, H, W))
        control_context.set_binding_shape(2, (1,))
        control_context.set_binding_shape(3, (1, 77, 768))
        self.model.control_context = control_context
        print("finished")
        """-----------------------------------------------"""

        """-----------------------------------------------转换diffusion_model为engine-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        diffusion_model = self.model.model.diffusion_model #找对了
        if not os.path.isfile("sd_diffusion_single_fp16.engine"):
            x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
            time_in = torch.zeros(1, dtype=torch.int64).to("cuda")
            context_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
            control = []
            control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            #diffusion_test = diffusion_model(x=x_in, timesteps=time_in, context=context_in, control=control)
            #print("diffusion_model的输出为:",diffusion_test.shape)
            input_names = ["x_in", "time_in", "context_in","crotrol"]
            output_names = ["out_h"]
            # dynamic_table = {"x_in" : {0 : "bs", 2 : "H", 3 : "W"},
            #                  "time_in" : {0 : "bs"},
            #                  "context_in" : {0 : "bs"},
            #                 }
            print("开始转换diffusion_model为onnx！\n")
            torch.onnx.export(diffusion_model,               
                                (x_in, time_in, context_in, control),  
                                "./sd_diffusion_single_fp16.onnx",   
                                export_params=True,#
                                opset_version=16,
                                do_constant_folding=True,
                                keep_initializers_as_inputs=True,
                                input_names =input_names, 
                                output_names = output_names)
            print("diffusion_model转换为onnx成功！\n")

            # print("开始转换diffusion_model为engine！\n")
            # #os.system("trtexec --onnx=sd_diffusion_fp16.onnx --saveEngine=sd_diffusion_fp16.engine --fp16")
            # print("转换diffusion_model为engine成功！\n")
        # print("开始加载diffusion_model的engine！\n")
        # with open("./sd_diffusion_fp16.engine", 'rb') as f:
        #     engine_str = f.read()
        # diffusion_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        # diffusion_context = diffusion_engine.create_execution_context()
        # diffusion_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
        # diffusion_context.set_binding_shape(1, (1,))
        # diffusion_context.set_binding_shape(2, (1, 77, 768))
        # diffusion_context.set_binding_shape(4, (1, 320, H//8, W //8))
        # diffusion_context.set_binding_shape(5,( 1, 320, H//8, W //8))
        # diffusion_context.set_binding_shape(6, (1, 320, H//8, W //8))
        # diffusion_context.set_binding_shape(7, (1, 320, H//16, W //16))
        # diffusion_context.set_binding_shape(8,(1, 640, H//16, W //16 ))
        # diffusion_context.set_binding_shape(9, (1, 640, H//16, W //16))
        # diffusion_context.set_binding_shape(10,( 1, 640, H//16, W //16))
        # diffusion_context.set_binding_shape(11, (1, 1280, H//32, W //32))
        # diffusion_context.set_binding_shape(12, (1,1280, H//32, W //32))
        # diffusion_context.set_binding_shape(13, (1, 1280, H//64, W //64))
        # diffusion_context.set_binding_shape(14, (1, 1280, H//64, W //64))
        # diffusion_context.set_binding_shape(15, (1, 1280, H//64, W //64))
        # diffusion_context.set_binding_shape(16, (1, 1280, H//64, W //64))
        #self.model.diffusion_context = diffusion_context
        print("加载diffusion_model的engine成功！\n")
        """-----------------------------------------------"""

        """----------------------------------------------转换cond_stage_model为engine-----------------"""
        first_stage_model = self.model.first_stage_model
        if not os.path.isfile("first_stage_fp16.engine"):
            
        """-----------------------------------------------"""

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution) #(256,384,3)
            H, W, C = img.shape 

            detected_map = self.apply_canny(img, low_threshold, high_threshold) #(256,384)
            detected_map = HWC3(detected_map) #(256,384,3)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0 #(256,384,3)
            control = torch.stack([control for _ in range(num_samples)], dim=0) #(1,256,384,3)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone() #(1,3,256,384)

            if seed == -1: #这里是随机种子
                seed = random.randint(0, 65535) 
            seed_everything(seed) #这里是设置随机种子

            if config.save_memory: #这里是设置是否保存内存
                self.model.low_vram_shift(is_diffusing=False) 
            
            """这里是重灾区"""
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]} #设置条件参数
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]} #这里用到encode来生成了？
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples, #dim_steps（DDIM算法的步数）、num_samples（生成的样本数量）、shape（样本的形状）、cond（条件）
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond) 
            """"""

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results
    
