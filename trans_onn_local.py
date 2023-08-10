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
        """----------------------------------------------转换cond_stage_model为onnx-----------------"""
        cond_stage_model = self.model.cond_stage_model
    
        clip = cond_stage_model.transformer #

        input_ids = torch.zeros((1,77),dtype=torch.int32).to("cuda")  #需要特别注意这里的输入是int64
        dynamic_axes = {'input_ids' : {0 : 'bs'},
                        'context' : {0 : 'bs'},
                        'pooled_output' : {0 : 'bs'}}
        input_names = ["input_ids"]
        output_names = ["context","pooled_output"]
        print("开始转换clip为onnx")
        torch.onnx.export(clip,
                            (input_ids),
                            "./models/onnxmodels/sd_clip_fp32-test-1326.onnx",
                        export_params=True,
                        opset_version=16,
                        do_constant_folding=True,
                        keep_initializers_as_inputs=True,
                        input_names = input_names, 
                        output_names = output_names,
                        dynamic_axes=dynamic_axes)
        print("clip转换完成")

        # os.system("trtexec --onnx=./models/onnxmodels/sd_clip_fp32-test-1326.onnx --saveEngine=./models/enginemodels/sd_clip_fp32-test-1326.plan --workspace=1000")
        """-----------------------------------------------"""

        """-----------------------------------------------转换control_model为onnx-----------------------------------------------"""
        control_model = self.model.control_model
        if not os.path.isfile("./models/onnxmodels/sd_control_fp16-test.onnx"):
            x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
            h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
            t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
            c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
            # controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in) #这里是一个测验
            output_names = []
            for i in range(13): #这里还不知道为什么是13,但确实是13个
                output_names.append("out_"+ str(i))
            # dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
            #                     'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
            #                     't_in' : {0 : 'bs'}, 
            #                     'c_in' : {0 : 'bs'}} #这里是需要看一下怎么写的
            # for i in range(13):
            #     dynamic_table[output_names[i]] = {0 : "bs"}
            print("开始转换control为onnx")
            torch.onnx.export(control_model,               
                                (x_in, h_in, t_in, c_in),  
                                "./models/onnxmodels/sd_control_fp16-test.onnx",   
                                export_params=True,
                                opset_version=16,
                                do_constant_folding=True,
                                keep_initializers_as_inputs=True,
                                input_names = ['x_in', "h_in", "t_in", "c_in"], 
                                output_names = output_names)
            print("control转换完成")
        """-----------------------------------------------"""

        """-----------------------------------------------转换diffusion_model为onnx-----------------------------------------------"""
        diffusion_model = self.model.model.diffusion_model #找对了
        if not os.path.isfile("./models/onnxmodels/sd_diffusion_fp16-test.onnx"):
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
                                "./models/onnxmodels/sd_diffusion_fp16-test.onnx",   
                                export_params=True,#
                                opset_version=16,
                                keep_initializers_as_inputs=True,
                                do_constant_folding=True,
                                input_names =input_names, 
                                output_names = output_names)
            print("转换diffusion_model为onnx成功！")
        """-----------------------------------------------"""

        """----------------------------------------------转换cond_stage_model为onnx-----------------"""
        first_stage_model = self.model.first_stage_model
        if not os.path.isfile("first_stage_fp16.onnx"):
            pass
        """-----------------------------------------------"""

        # 建议将TensorRT的onnx存到一个dict中，然后将dict给下面的DDIMSampler做初始化
        # 例如self.onnx = {"clip": xxx_onnx, "control_net": xxx_onnx, ...}
        #self.ddim_sampler = DDIMSampler(self.model, onnx=self.onnx)
        # 最后，将DDIMSampler中调用pytorch4个子模型操作的部分，用onnx推理代替，工作就做完了。

if __name__ == "__main__":
    h = hackathon()
    h.initialize()
