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

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from cuda import cudart
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
   


        H = 256
        W = 384
    
        ####################################### 配置clip ############################################
        # with open("./models/enginemodels/sd_clip_fp32.plan", 'rb') as f:
        #     clip_engine_str = f.read()
        # clip_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(clip_engine_str)
        # clip_context = clip_engine.create_execution_context()
        # clip_tensor_name = [clip_engine.get_tensor_name(i) for i in range(3)]
        # # clip的输入
        # self.model.cond_stage_model.token=torch.zeros((1,77),dtype=torch.int32).to("cuda")       
        # clip_context.set_tensor_address(clip_tensor_name[0], self.model.cond_stage_model.token)  
        # # clip的输出
        # self.model.cond_stage_model.last_hidden_state=torch.zeros((1,77,768),dtype=torch.float32).to("cuda")
        # clip_context.set_tensor_address(clip_tensor_name[1], self.model.cond_stage_model.last_hidden_state) 
        # self.model.cond_stage_model.pooler_output=torch.zeros((1,768),dtype=torch.float32).to("cuda")
        # clip_context.set_tensor_address(clip_tensor_name[2], self.model.cond_stage_model.pooler_output)
        # # CUDA Graph capture
        # _, clip_stream = cudart.cudaStreamCreate()
        # cudart.cudaStreamBeginCapture(clip_stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        # clip_context.execute_async_v3(clip_stream)
        # _, clip_graph = cudart.cudaStreamEndCapture(clip_stream)
        # _, clip_graphExe = cudart.cudaGraphInstantiate(clip_graph, 0)
        # self.model.cond_stage_model.clip_graphExe=clip_graphExe
        # self.model.cond_stage_model.clip_graphExe=clip_graphExe
        # # in clip do inference with CUDA graph
        # cudart.cudaGraphLaunch(self.clip_graphExe, self.clip_stream)
        # cudart.cudaStreamSynchronize(self.stream)


        ################################### 配置controlnet ############################################
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        with open("./models/enginemodels/sd_control_fp16.engine", 'rb') as f:
            control_engine_str = f.read()
        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(control_engine_str)
        control_context = control_engine.create_execution_context()
        control_nIO = control_engine.num_io_tensors
        control_tensor_name = [control_engine.get_tensor_name(i) for i in range(control_nIO)]
        # control的输入
        self.model.x_noisy = torch.zeros((1, 4, H//8, W//8),dtype=torch.float32).to("cuda")  
        self.model.hint_in = torch.zeros((1, 3, H, W),dtype=torch.float32).to("cuda")  
        self.model.t = torch.zeros((1),dtype=torch.int32).to("cuda")  #int32可能有影响
        self.model.cond_txt = torch.zeros((1,77,768),dtype=torch.float32).to("cuda") 
        b, c, h, w = self.model.x_noisy.shape
        buffer_device = []
        buffer_device.append(self.model.x_noisy.reshape(-1).data_ptr())
        buffer_device.append(self.model.hint_in.reshape(-1).data_ptr())
        buffer_device.append(self.model.t.reshape(-1).data_ptr())
        buffer_device.append(self.model.cond_txt.reshape(-1).data_ptr())
        # control的输出
        control_out = []
        self.model.control_out_1 = torch.zeros(b, 320, h, w, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_1.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_1.reshape(-1).data_ptr())
        self.model.control_out_2 = torch.zeros(b, 320, h, w, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_2.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_2.reshape(-1).data_ptr())
        self.model.control_out_3 = torch.zeros(b, 320, h, w, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_3.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_3.reshape(-1).data_ptr())
        self.model.control_out_4 = torch.zeros(b, 320, h//2, w//2, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_4.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_4.reshape(-1).data_ptr())
        self.model.control_out_5 = torch.zeros(b, 640, h//2, w//2, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_5.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_5.reshape(-1).data_ptr())
        self.model.control_out_6 = torch.zeros(b, 640, h//2, w//2, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_6.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_6.reshape(-1).data_ptr())
        self.model.control_out_7 = torch.zeros(b, 640, h//4, w//4, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_7.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_7.reshape(-1).data_ptr())
        self.model.control_out_8 = torch.zeros(b, 1280, h//4, w//4, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_8.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_8.reshape(-1).data_ptr())
        self.model.control_out_9 = torch.zeros(b, 1280, h//4, w//4, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_9.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_9.reshape(-1).data_ptr())
        self.model.control_out_10 = torch.zeros(b, 1280, h//8, w//8, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_10.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_10.reshape(-1).data_ptr())
        self.model.control_out_11 = torch.zeros(b, 1280, h//8, w//8, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_11.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_11.reshape(-1).data_ptr())
        self.model.control_out_12 = torch.zeros(b, 1280, h//8, w//8, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_12.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_12.reshape(-1).data_ptr())
        self.model.control_out_13 = torch.zeros(b, 1280, h//8, w//8, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.control_out_13.reshape(-1).data_ptr())
        control_out.append(self.model.control_out_13.reshape(-1).data_ptr())
        for i in range(control_nIO):
            control_context.set_tensor_address(control_tensor_name[i], buffer_device[i])
        self.model.control_context = control_context
        # CUDA Graph capture
        # _, control_stream = cudart.cudaStreamCreate()
        # cudart.cudaStreamBeginCapture(control_stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        # control_context.execute_async_v3(control_stream)  #在捕获的时候，异步执行
        # cudart.cudaStreamSynchronize(control_stream) 
        # _, control_graph = cudart.cudaStreamEndCapture(control_stream)
        # _, control_graphExe = cudart.cudaGraphInstantiate(control_graph, 0)
        # self.model.control_graphExe=control_graphExe
        # self.model.control_stream=control_stream
        ####################################### 配置unet ##############################################
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')        
        with open("./models/enginemodels/sd_diffusion_fp16.engine", 'rb') as f:
            diffusion_engine_str = f.read()
        diffusion_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(diffusion_engine_str)
        diffusion_context = diffusion_engine.create_execution_context()

        diffusion_nIO = diffusion_engine.num_io_tensors
        diffusion_tensor_name = [diffusion_engine.get_tensor_name(i) for i in range(diffusion_nIO)]
        # diffusion的输入
        buffer_device = []
        buffer_device.append(self.model.x_noisy.reshape(-1).data_ptr())
        buffer_device.append(self.model.t.reshape(-1).data_ptr())
        buffer_device.append(self.model.cond_txt.reshape(-1).data_ptr())
        for co in control_out:
            buffer_device.append(co)
        # diffusion的输出
        self.model.eps = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to("cuda")
        buffer_device.append(self.model.eps.reshape(-1).data_ptr())
        for i in range(control_nIO):
            diffusion_context.set_tensor_address(diffusion_tensor_name[i], buffer_device[i])
        self.model.diffusion_context = diffusion_context
        # CUDA Graph capture
        # _, diffusion_stream = cudart.cudaStreamCreate()
        # cudart.cudaStreamBeginCapture(diffusion_stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        # diffusion_context.execute_async_v3(diffusion_stream)
        # cudart.cudaStreamSynchronize(control_stream) 
        # _, diffusion_graph = cudart.cudaStreamEndCapture(diffusion_stream)
        # _, diffusion_graphExe = cudart.cudaGraphInstantiate(diffusion_graph, 0)
        # self.model.diffusion_graphExe=diffusion_graphExe
        # self.model.diffusion_stream=diffusion_stream


        ######################################### 配置decoder #########################################

        print("finished")



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
            
            ddim_steps = 20
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

# if __name__=="__main__":
#     h = hackathon()
#     h.initialize()