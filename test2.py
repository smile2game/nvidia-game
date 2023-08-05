import os
import numpy as np
import tensorrt as trt
from cuda import cudart
import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
H = 256
W = 384
#trtexec --onnx=./models/onnxmodels/sd_clip_fp16-test-2131.onnx --saveEngine=./models/enginemodels/sd_clip_fp16-test-2131.engine --verbose=True 

trtFile = "./models/enginemodels/sd_clip_fp32-test-1326.plan"
data = torch.zeros(1, 77, dtype=torch.int64).to('cuda')


def run():
    logger = trt.Logger(trt.Logger.VERBOSE)  
    with open(trtFile, "rb") as f:
            engineString = f.read()

    #生成推理引擎
    engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)                                    
    context = engine.create_execution_context()
    buffer_device = []
    buffer_device.append(data.reshape(-1).data_ptr())

    output1 = torch.randn(1, 77,768,dtype=torch.float32).cuda()
    output2 = torch.randn(1, 768,dtype=torch.float32).cuda()
    buffer_device.append(output1.reshape(-1).data_ptr())
    buffer_device.append(output2.reshape(-1).data_ptr())
    context.execute_v2(buffer_device)
    print(output1)
    print(output2)
if __name__ == "__main__":
    run()
    