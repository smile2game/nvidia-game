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
from torchsummary import summary
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
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        control_model = self.model.control_model
        if not os.path.isfile("./middle_model/sd_control_fp16-test.engine"):
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
            torch.onnx.export(control_model,               
                                (x_in, h_in, t_in, c_in),  
                                "./sd_control_test.onnx",   
                                export_params=True,
                                opset_version=16,
                                do_constant_folding=True,
                                keep_initializers_as_inputs=True,
                                input_names = ['x_in', "h_in", "t_in", "c_in"], 
                                output_names = output_names)
                                # dynamic_axes = dynamic_table)
            os.system("trtexec --onnx=sd_control_test.onnx --saveEngine=sd_control_fp16.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")
        with open("./sd_control_fp16.engine", 'rb') as f:
            engine_str = f.read()
        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        control_context = control_engine.create_execution_context()
        # control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
        # control_context.set_binding_shape(1, (1, 3, H, W))
        # control_context.set_binding_shape(2, (1,))
        # control_context.set_binding_shape(3, (1, 77, 768))
        self.model.control_context = control_context
        print("controlnet成功转为engine模型")
        """-----------------------------------------------"""

        """-----------------------------------------------转换diffusion_model为engine-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        diffusion_model = self.model.model.diffusion_model #找对了
        if not os.path.isfile("sd_diffusion_fp16-test.engine"):
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
                                "./sd_diffusion_fp16.onnx",   
                                export_params=True,#
                                opset_version=16,
                                keep_initializers_as_inputs=True,
                                do_constant_folding=True,
                                input_names =input_names, 
                                output_names = output_names)
            print("diffusion_model转换为onnx成功！\n")

            print("开始转换diffusion_model为engine！\n")
            os.system("trtexec --onnx=sd_diffusion_fp16.onnx --saveEngine=sd_diffusion_fp16.engine --fp16")
            print("转换diffusion_model为engine成功！\n")
        print("开始加载diffusion_model的engine！\n")
        with open("./sd_diffusion_fp16.engine", 'rb') as f:
            engine_str = f.read()
        diffusion_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        diffusion_context = diffusion_engine.create_execution_context()
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
        self.model.diffusion_context = diffusion_context
        print("加载diffusion_model的engine成功！\n")
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
