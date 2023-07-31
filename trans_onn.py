from share import *
import os
import config
import gc
import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import tensorrt as trt

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
        control_model = self.model.control_model
        x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
        h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
        t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
        c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
        output_names = []
        for i in range(13): #这里还不知道为什么是13,但确实是13个
            output_names.append("out_"+ str(i))

        torch.onnx.export(control_model,               
                            (x_in, h_in, t_in, c_in),  
                            "./sd_control_fp16-test.onnx",   
                            export_params=True,
                            opset_version=16,
                            do_constant_folding=True,
                            keep_initializers_as_inputs=True,
                            input_names = ['x_in', "h_in", "t_in", "c_in"], 
                            output_names = output_names)
                            # dynamic_axes = dynamic_table)
        # --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768
        print("controlnet成功转为engine模型")
        """-----------------------------------------------"""

        """-----------------------------------------------转换diffusion_model为engine-----------------------------------------------"""
        diffusion_model = self.model.model.diffusion_model #找对了
        
        print("转换diffusion_model为onnx模型")
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
        input_names = ["x_in", "time_in", "context_in","crotrol"]
        output_names = ["out_h"]
        
        print("开始转换diffusion_model为onnx！\n")
        torch.onnx.export(diffusion_model,               
                            (x_in, time_in, context_in, control),  
                            "./sd_diffusion_fp16-test.onnx",   
                            export_params=True,#
                            opset_version=16,
                            keep_initializers_as_inputs=True,
                            do_constant_folding=True,
                            input_names =input_names, 
                            output_names = output_names)
        print("转换diffusion_model为onnx成功！")
        """-----------------------------------------------"""

        """----------------------------------------------转换cond_stage_model为engine-----------------"""
        first_stage_model = self.model.first_stage_model
        if not os.path.isfile("first_stage_fp16.engine"):
            pass
        """-----------------------------------------------"""

        # 建议将TensorRT的engine存到一个dict中，然后将dict给下面的DDIMSampler做初始化
        # 例如self.engine = {"clip": xxx_engine, "control_net": xxx_engine, ...}
        #self.ddim_sampler = DDIMSampler(self.model, engine=self.engine)
        # 最后，将DDIMSampler中调用pytorch4个子模型操作的部分，用engine推理代替，工作就做完了。

if __name__ == "__main__":
    h = hackathon()
    h.initialize()
