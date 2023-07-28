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
        self.model = create_model('./models/cldm_v15.yaml').cpu()#创建模型
        self.model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        self.state_dict = {
            "clip" : "cond_stage_model",
            "control_net" : "control_model",
            "unet" : "diffusion_model",
            "vae" : "first_stage_model"
        }
        H = 256
        W = 384

        for k,v in self.state_dict.items(): #获取子模型
            # if k!="unet":
            #     temp_model = getattr(self.model, v)
            #     print("这里的k是:",k)
            #     print("得到临时模型:\n",temp_model)
            # else:
            #     temp_model = getattr(self.model.model, v) 
            print(k,v,"\n")
            if k == "clip":
                pass
                # """
                # 获取各个参数
                # """
                # model = temp_model.transformer #获取transformer导出为onnx
                # self.tokenizer = temp_model.tokenizer #获取tokenizer
                # onnx_path = "./bmodels/clip-onnx"
                # """这里需要debug"""
                # example_input = torch.randn(1, 1, 256, 384).cuda() #这个还需要debug一下
                # input_names = ["input_tensor"] #输入输出名列表，不是这样子随便写的，需要搞明白这个模型的输入输出
                # output_names = ["output_tensor"]
                # #输入变量`input_ids`和输出变量`text_embeddings`的第0维可以动态变化，其实也就是batch_size支持动态咯
                # dynamic_axes = {'input_ids': {0: 'B'},'text_embeddings': {0: 'B'}} 
                # opset_version = 17 #onnx版本
                # print("在导出之前都还一切正常！")
                # #开始导出
                # torch.onnx.export(model, 
                #                 example_input,
                #                 onnx_path, 
                #                 input_names=input_names, 
                #                 output_names=output_names, 
                #                 dynamic_axes=dynamic_axes,
                #                 opset_version=17)
                


                # print("成功导出onnx模型")
            elif k == "control_net":
                if not os.path.isfile("./models/onnxmodels/sd_control_test.onnx"):
                    print("开始导出control_net的onnx模型")
                    control_model = self.model.control_model  #获取control_model
                    x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda") #获取输入尺寸
                    h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda") #获取输入
                    t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
                    c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
                    controls = control_model(x = x_in,hint = h_in,timesteps = t_in,context = c_in)
                    output_names = []
                    for i in range(13):
                        output_names.append("output_"+str(i))
                    dynamic_table = {'x_in' : {0:"bs",2:"H",3:"W"},
                                    'h_in' : {0 : 'bs',2 : '8H',3: '8W'},
                                    't_in' : {0:'bs'},
                                    'c_in' : {0:'bs'}}   
                    for i in range(13):
                        dynamic_table[output_names[i]] = {0 : "bs"}
                    torch.onnx.export(control_model,
                                    (x_in,h_in,t_in,c_in),
                                    "./models/onnxmodels/sd_control_test.onnx",
                                    export_params=True,
                                    opset_version=17,
                                    do_constant_folding=True,
                                    keep_initializers_as_inputs=True,
                                    input_names=['x_in','h_in','t_in','c_in'],
                                    output_names=output_names,
                                    dynamic_axes=dynamic_table)
                    print("成功导出control_net的onnx模型")
                if not os.path.isfile("sd_control_fp16.engine"):
                    print("开始导出control_net的engine模型")
                    os.system("trtexec --onnx=./models/onnxmodels/sd_control_test.onnx --saveEngine=./models/enginemodels/sd_control_fp16.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")
                    print("成功导出control_net的engine模型")
                with open("./sd_control_fp16.engine", 'rb') as f:
                    engine_str = f.read()

                control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
                control_context = control_engine.create_execution_context()

                control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
                control_context.set_binding_shape(1, (1, 3, H, W))
                control_context.set_binding_shape(2, (1,))
                control_context.set_binding_shape(3, (1, 77, 768))
                self.model.control_context = control_context
                print("完成engine生成和绑定")


            elif k == "unet":
                pass
            else:
                model = None

        # 建议将TensorRT的engine存到一个dict中，然后将dict给下面的DDIMSampler做初始化
        # 例如self.engine = {"clip": xxx_engine, "control_net": xxx_engine, ...}
        #self.ddim_sampler = DDIMSampler(self.model, engine=self.engine)
        # 最后，将DDIMSampler中调用pytorch4个子模型操作的部分，用engine推理代替，工作就做完了。

if __name__ == "__main__":
    h = hackathon()
    h.initialize()
