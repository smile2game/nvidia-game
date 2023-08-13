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

import onnx
from polygraphy.backend.onnx import onnx_from_path

class hackathon():
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        H = 256
        W = 384
        """-----------------------------------------------加载clip的engine模型-----------------------------------------------"""
       

        """---------------------------加载controlnet--------------------"""
        if not os.path.isfile("sd_control_fp16.engine"):
            control_model = self.model.control_model 
            x_in = torch.randn(2, 4, H//8, W //8, dtype=torch.float32).to("cuda")
            h_in = torch.randn(2, 3, H, W, dtype=torch.float32).to("cuda")
            t_in = torch.zeros(2, dtype=torch.int64).to("cuda")
            c_in = torch.randn(2, 77, 768, dtype=torch.float32).to("cuda")
            # controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)
            output_names = []
            for i in range(13):
                output_names.append("out_"+ str(i))

            dynamic_table = {'x_in' : {0 : 'bs'}, 
                                'h_in' : {0 : 'bs'}, 
                                't_in' : {0 : 'bs'},
                                'c_in' : {0 : 'bs'}}
            for i in range(13):
                dynamic_table[output_names[i]] = {0 : "bs"}

            torch.onnx.export(control_model,               
                                (x_in, h_in, t_in, c_in),  
                                "./sd_control.onnx",   
                                export_params=True,
                                opset_version=17,
                                do_constant_folding=True,
                                keep_initializers_as_inputs=True,
                                input_names = ['x_in', "h_in", "t_in", "c_in"], 
                                output_names = output_names,
                                dynamic_axes = dynamic_table)

            os.system("trtexec --onnx=./sd_control.onnx --saveEngine=./sd_control_fp16.engine --fp16 --verbose --useCudaGraph --optShapes=x_in:2x4x32x48,h_in:2x3x256x384,t_in:2,c_in:2x77x768  --minShapes=x_in:2x4x32x48,h_in:2x3x256x384,t_in:2,c_in:2x77x768  --maxShapes=x_in:2x4x32x48,h_in:2x3x256x384,t_in:2,c_in:2x77x768  --builderOptimizationLevel=3")
            #自带cuda graph

        with open("./sd_control_fp16.engine", 'rb') as f:
            engine_str = f.read()
        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        control_context = control_engine.create_execution_context()
        control_context.set_binding_shape(0, (2, 4, H // 8, W // 8))
        control_context.set_binding_shape(1, (2, 3, H, W))
        control_context.set_binding_shape(2, (2,))
        control_context.set_binding_shape(3, (2, 77, 768))
        self.model.control_context = control_context
        print("\ncontrolnet成功启用")

        """-----------------------------------------------加载unet的engine模型-----------------------------------------------"""
        if not os.path.isfile("sd_diffusion_fp16.engine"):
            diffusion_model = self.model.model.diffusion_model #找对了
            print("转换diffusion_model为onnx模型")
            x_in = torch.randn(2, 4, H//8, W //8, dtype=torch.float32).to("cuda")
            time_in = torch.zeros(2, dtype=torch.int64).to("cuda")
            context_in = torch.randn(2, 77, 768, dtype=torch.float32).to("cuda")
            control = []
            control.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
            control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))

            input_names = ["x_in", "time_in", "context_in"]
            for i in range(13):
                input_names.append("control_"+str(i))
            output_names = ["out_h"]

            dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                                'time_in' : {0 : 'bs'},
                                'context_in' : {0 : 'bs'}}
            for i in range(3,15):
                    dynamic_table[input_names[i]] = {0 : "bs"}


            print("开始转换diffusion_model为onnx！\n")
            torch.onnx.export(diffusion_model,               
                                (x_in, time_in, context_in, control),  
                                "./ooo/sd_diffusion.onnx",   
                                export_params=True,#
                                opset_version=17,
                                keep_initializers_as_inputs=True,
                                do_constant_folding=True,
                                input_names =input_names, 
                                output_names = output_names)
            
            #dynamic

            print("转换diffusion_model为onnx成功！")

            os.system("trtexec --onnx=./ooo/sd_diffusion.onnx --saveEngine=./sd_diffusion_fp16.engine --fp16 --useCudaGraph --verbose --builderOptimizationLevel=3")
            #level = 4 会 killed; level = 5 会 segment default

        with open("sd_diffusion_fp16.engine", 'rb') as f:
            diffusion_engine_str = f.read()
        diffusion_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(diffusion_engine_str)
        diffusion_context = diffusion_engine.create_execution_context()

        diffusion_context.set_binding_shape(0, (2, 4, H//8, W //8))
        diffusion_context.set_binding_shape(1, (2,))
        diffusion_context.set_binding_shape(2, (2, 77,768))

        diffusion_context.set_binding_shape(3, (2, 320, H//8, W //8))
        diffusion_context.set_binding_shape(4, (2, 320, H//8, W //8))
        diffusion_context.set_binding_shape(5, (2, 320, H//8, W //8))

        diffusion_context.set_binding_shape(6, (2, 320, H//16, W //16))
        diffusion_context.set_binding_shape(7, (2, 640, H//16, W //16))
        diffusion_context.set_binding_shape(8, (2, 640, H//16, W //16))
        diffusion_context.set_binding_shape(9, (2, 640, H//32, W //32))

        diffusion_context.set_binding_shape(10, (2,1280, H//32, W //32))
        diffusion_context.set_binding_shape(11, (2,1280, H//32, W //32))
        diffusion_context.set_binding_shape(12, (2,1280, H//64, W //64))
        diffusion_context.set_binding_shape(13, (2,1280, H//64, W //64))
        diffusion_context.set_binding_shape(14, (2,1280, H//64, W //64))
        diffusion_context.set_binding_shape(15, (2,1280, H//64, W //64))
        
        self.model.diffusion_context = diffusion_context
        print("加载成功diffusion_model的engine")
        """----------------------------------------------------------------------------------------------"""

        """------------------------添加vae的部分-----------------------"""
        if not os.path.isfile("sd_vae_fp16.engine"):
            model = self.model.first_stage_model
            # vae调用的是decode,而导出onnx需要forward,所以这里做一个替换即可。
            model.forward = model.decode
            print("开始生成vae的onnx")
            torch.onnx.export(
                model,
                (torch.randn(1, 4, 32, 48, device="cuda")),
                './sd_vae.onnx',
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['z'],
                output_names=['dec'],
                dynamic_axes={'z': {0: 'B'}, 'dec': {0: 'B'}},
            )
            print("vae的onnx生成完成")
            os.system("trtexec --onnx=./sd_vae.onnx --saveEngine=./sd_vae_fp16.engine --fp16 --useCudaGraph --optShapes=z:1x4x32x48 --builderOptimizationLevel=5")
            
        with open("./sd_vae_fp16.engine", 'rb') as f:
            engine_str = f.read()
        vae_decode_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        vae_decode_context = vae_decode_engine.create_execution_context()
        vae_decode_context.set_binding_shape(0, (1, 4, 32,48)) 
        self.model.vae_decode_context = vae_decode_context
        print("finished vae!")
        """-----------------------------------------------"""

        """-------------------------提前开buffer----------------------"""
        #controlnet:4 -> 13
        #unet: 3 + 13 -> 1
        #总共：4 +13 +3+1=21
        self.model.control_out = []

        self.model.control_out.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.control_out.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.model.eps = torch.zeros(2, 4, 32, 48, dtype=torch.float32).to("cuda")
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
            ddim_steps = 10  #当小于6的时候，图像会发生质变
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
    
if __name__ == "__main__":
    h = hackathon()
    h.initialize()