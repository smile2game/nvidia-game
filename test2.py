import os
import numpy as np
import tensorrt as trt
from cuda import cudart
import torch
import datetime
# os.environ['DISABLE_STREAM_CAPTURING'] = 0

H = 256
W = 384

trtFile = "./models/enginemodels/sd_diffusion_fp16-test.engine"
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

def run():
    logger = trt.Logger(trt.Logger.VERBOSE)  
    trt.init_libnvinfer_plugins(logger, '')
    with open(trtFile, "rb") as f:
            engineString = f.read()
    #生成推理引擎)
    diffusion_engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)              
    
    diffusion_context = diffusion_engine.create_execution_context()

    buffer_device = [] #走的这里
    buffer_device.append(x_in.reshape(-1).data_ptr())
    buffer_device.append(time_in.reshape(-1).data_ptr())
    buffer_device.append(context_in.reshape(-1).data_ptr())
    for temp in control:
        buffer_device.append(temp.reshape(-1).data_ptr())
    eps = torch.zeros(1, 4, 32, 48, dtype=torch.float32).to("cuda")
    buffer_device.append(eps.reshape(-1).data_ptr())
    start = datetime.datetime.now().timestamp()
    diffusion_context.execute_v2(buffer_device)
    end = datetime.datetime.now().timestamp()
    print("\ndiffusion消耗时间为：", (end - start)*1000 )
    print(eps)

if __name__ == "__main__":
    run()
    