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

diffusion_trt = "./1-engine_b2/sd_diffusion_fp16.engine"

class hackathon():
    def initialize(self):
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
     ########################################################
        with open(diffusion_trt, "rb") as f:
                engineString = f.read()
        #生成推理引擎)
        diffusion_engine = trt.Runtime(self.logger).deserialize_cuda_engine(engineString)              
        self.diffusion_context = diffusion_engine.create_execution_context()
        # self.diffusion_context.set_binding_shape(0, (2, 4, H//8, W //8))
        # self.diffusion_context.set_binding_shape(1, (2,))
        # self.diffusion_context.set_binding_shape(2, (2, 77,768))

        # self.diffusion_context.set_binding_shape(3, (2, 320, H//8, W //8))
        # self.diffusion_context.set_binding_shape(4, (2, 320, H//8, W //8))
        # self.diffusion_context.set_binding_shape(5, (2, 320, H//8, W //8))

        # self.diffusion_context.set_binding_shape(6, (2, 320, H//16, W //16))
        # self.diffusion_context.set_binding_shape(7, (2, 640, H//16, W //16))
        # self.diffusion_context.set_binding_shape(8, (2, 640, H//16, W //16))
        # self.diffusion_context.set_binding_shape(9, (2, 640, H//32, W //32))

        # self.diffusion_context.set_binding_shape(10, (2,1280, H//32, W //32))
        # self.diffusion_context.set_binding_shape(11, (2,1280, H//32, W //32))
        # self.diffusion_context.set_binding_shape(12, (2,1280, H//64, W //64))
        # self.diffusion_context.set_binding_shape(13, (2,1280, H//64, W //64))
        # self.diffusion_context.set_binding_shape(14, (2,1280, H//64, W //64))
        # self.diffusion_context.set_binding_shape(15, (2,1280, H//64, W //64))
        
        diffusion_nIO = diffusion_engine.num_io_tensors
        diffusion_tensor_name = [diffusion_engine.get_tensor_name(i) for i in range(diffusion_nIO)]
        ############################创建输入输出buffer
        start = datetime.datetime.now().timestamp()

        self.x_in = torch.randn(2, 4, H//8, W //8, dtype=torch.float32).to("cuda")
        self.time_in = torch.zeros(2, dtype=torch.int32).to("cuda")
        self.context_in = torch.randn(2, 77, 768, dtype=torch.float32).to("cuda")
        self.control = []
        self.control.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 320, H//8, W //8, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 320, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 640, H//16, W //16, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 640, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 1280, H//32, W //32, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))
        self.control.append(torch.randn(2, 1280, H//64, W //64, dtype=torch.float32).to("cuda"))

        #buffer处理
        buffer_device = []
        buffer_device.append(self.x_in.reshape(-1).data_ptr())
        buffer_device.append(self.time_in.reshape(-1).data_ptr())
        buffer_device.append(self.context_in.reshape(-1).data_ptr())
        for temp in self.control:
            buffer_device.append(temp.reshape(-1).data_ptr())
        self.eps = torch.zeros(2, 4, 32, 48, dtype=torch.float32).to("cuda")
        buffer_device.append(self.eps.reshape(-1).data_ptr())

        #创流 开捕
        for i in range(diffusion_nIO):
            self.diffusion_context.set_tensor_address(diffusion_tensor_name[i], buffer_device[i])
        self.diffusion_context.execute_async_v3(diffusion_engine)
        cudart.cudaStreamSynchronize(diffusion_stream)

        _, diffusion_stream = cudart.cudaStreamCreate()
        cudart.cudaStreamBeginCapture(diffusion_stream, cudart.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal)
        for i in range(diffusion_nIO):
            self.diffusion_context.set_tensor_address(diffusion_tensor_name[i], buffer_device[i])
        self.diffusion_context.execute_async_v3(diffusion_engine)
        cudart.cudaStreamSynchronize(diffusion_stream)
        
        
        #结束捕获
        # _, graph = cudart.cudaStreamEndCapture(diffusion_stream)
        # _, graphExe = cudart.cudaGraphInstantiate(graph, 0)
        end = datetime.datetime.now().timestamp()
        print("\n通过initialize节约的时间为:",(end-start)*1000)

      


    def process(self):
        ###################################        
        # _, diffusion_stream = cudart.cudaStreamCreate()
        # #execute
        # self.diffusion_context.execute_async_v3(diffusion_stream)
        self.diffusion_context.execute_async_v3(0)
        return self.eps
    
if __name__ == "__main__":
    times = []
    h = hackathon()
    h.initialize()
    start = datetime.datetime.now().timestamp()
    res = h.process()
    end = datetime.datetime.now().timestamp()
    times.append((end - start)*1000)
    # print("\ncontrolnet的输出为：",h.control_out)
    print("\ndiffusion的输出为：",res)
    print("\n执行process流程,消耗时间为：", times)
        