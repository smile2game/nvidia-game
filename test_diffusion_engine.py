import os
import numpy as np
import tensorrt as trt
from cuda import cudart
import torch
import datetime
from cuda import cudart

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

H = 256
W = 384

control_trt = "./models/enginemodels/sd_control_fp16.engine"
diffusion_trt = "./models/enginemodels/sd_diffusion_fp16.engine"

class hackathon():
    def initialize(self):
        self.logger = trt.Logger(trt.Logger.VERBOSE)  
        trt.init_libnvinfer_plugins(self.logger, '')
        ####################
        with open(control_trt, "rb") as f:
            engineString = f.read()
        #生成推理引擎)``
        self.control_engine = trt.Runtime(self.logger).deserialize_cuda_engine(engineString)              
        self.control_context = self.control_engine.create_execution_context()
        #################
        with open(diffusion_trt, "rb") as f:
                engineString = f.read()
        #生成推理引擎)
        self.diffusion_engine = trt.Runtime(self.logger).deserialize_cuda_engine(engineString)              
        self.diffusion_context = self.diffusion_engine.create_execution_context()
        
        ############################创建输入输出buffer
        start = datetime.datetime.now().timestamp()
        self.buffer_device = []
        b, c, h, w = 1,4,H//8,W//8
        self.control_out = []
        self.control_out.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control_out.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        
        self.buffer_device_diffsuion = []
        self.eps = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to("cuda")
        self.buffer_device.append(self.eps.reshape(-1).data_ptr())
        end = datetime.datetime.now().timestamp()
        print("\n通过initialize节约的时间为:",(end-start)*1000)

    def process(self):
        ###################################
        x_noisy = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
        hint_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
        t = torch.zeros(1, dtype=torch.int64).to("cuda")
        cond_txt = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
        self.buffer_device.append(x_noisy.reshape(-1).data_ptr())
        self.buffer_device.append(hint_in.reshape(-1).data_ptr())
        self.buffer_device.append(t.reshape(-1).data_ptr())
        self.buffer_device.append(cond_txt.reshape(-1).data_ptr())
        #execute
        self.control_context.execute_v2(self.buffer_device)
        ###################################
        self.x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
        self.time_in = torch.zeros(1, dtype=torch.int64).to("cuda")
        self.context_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
        self.control = []
        self.control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(1, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        
        self.buffer_device.append(self.x_in.reshape(-1).data_ptr())
        
        self.buffer_device.append(self.context_in.reshape(-1).data_ptr())
        self.buffer_device.append(self.time_in.reshape(-1).data_ptr())
        for temp in self.control:
            self.buffer_device.append(temp.reshape(-1).data_ptr())
        
        #execute
        self.diffusion_context.execute_v2(self.buffer_device)
        return self.eps
    
if __name__ == "__main__":
    times = []
    h = hackathon()
    h.initialize()
    start = datetime.datetime.now().timestamp()
    h.process()
    end = datetime.datetime.now().timestamp()
    times.append((end - start)*1000)
    # print("\ncontrolnet的输出为：",h.control_out)
    print("\ndiffusion的输出为：",h.eps)
    print("\n执行process流程,消耗时间为：", times)
        