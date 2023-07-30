from share import *
import config
import datetime
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
        # if not os.path.isfile("sd_cond_fp16.engine"):
        #     # x_in = torch.randn(1, 4, H, W, dtype=torch.float32).to("cuda")
        #     # h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
        #     # t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
        #     # c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
        #     batch_size = 2
        #     sequence_length = 15
        #     vocab_size = 49408  # 假设词汇表大小为 49408
        #     # 创建示例输入张量
        #     example_input = torch.randint(low=0, high=vocab_size, size=(batch_size, sequence_length), dtype=torch.long)
        #     encodes = cond_stage_model(example_input)
        """-----------------------------------------------"""




        """-----------------------------------------------转换control_model为engine-----------------------------------------------"""
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        control_model = self.model.control_model
        # if not os.path.isfile("sd_control_fp16.engine"):
        #     x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
        #     h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
        #     t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
        #     c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
        #     controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)
        #     output_names = []
        #     for i in range(13): #这里还不知道为什么是13
        #         output_names.append("out_"+ str(i))
        #     dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
        #                         'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
        #                         't_in' : {0 : 'bs'}, 
        #                         'c_in' : {0 : 'bs'}} #这里是需要看一下怎么写的
        #     for i in range(13):
        #         dynamic_table[output_names[i]] = {0 : "bs"}
        #     torch.onnx.export(control_model,               
        #                         (x_in, h_in, t_in, c_in),  
        #                         "./sd_control_test.onnx",   
        #                         export_params=True,
        #                         opset_version=16,
        #                         do_constant_folding=True,
        #                         keep_initializers_as_inputs=True,
        #                         input_names = ['x_in', "h_in", "t_in", "c_in"], 
        #                         output_names = output_names, 
        #                         dynamic_axes = dynamic_table)
        #     os.system("trtexec --onnx=sd_control_test.onnx --saveEngine=sd_control_fp16.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")
        with open("./sd_control_fp16.engine", 'rb') as f:
            engine_str = f.read()
        control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
        control_context = control_engine.create_execution_context()
        control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
        control_context.set_binding_shape(1, (1, 3, H, W))
        control_context.set_binding_shape(2, (1,))
        control_context.set_binding_shape(3, (1, 77, 768))
        self.model.control_context = control_context
        print("finished")


        """-----------------------------------------------"""

        """-----------------------------------------------转换diffusion_model为engine-----------------------------------------------"""
        diffusion_model = self.model.model.diffusion_model #找对了
        # if not os.path.isfile("sd_unet_fp16.engine"):
        #     # x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
        #     # h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
        #     # t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
        #     # c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
        #     batch_size = 2
        #     channels = 3
        #     height = 256
        #     width = 256
        #     timesteps = 4  # 假设有4个时间步

        #     # 创建示例输入张量
        #     example_input = torch.randn(batch_size, channels, height, width)
        #     example_timesteps = torch.arange(timesteps).unsqueeze(0)  # 假设时间步的张量

        #     # 加载模型

        #     # 导出模型为ONNX格式
        #     output_path = "path_to_save_model.onnx"
        #     torch.onnx.export(
        #         diffusion_model,
        #         (example_input, example_timesteps),
        #         output_path,
        #         opset_version=16,  # 根据你的需要进行调整
        #         do_constant_folding=True
        #     )

        #     print("ONNX模型导出成功！")
        """-----------------------------------------------"""

        """----------------------------------------------转换cond_stage_model为engine-----------------"""
        first_stage_model = self.model.first_stage_model
        if not os.path.isfile("first_stage_fp16.engine"):
            pass
        """-----------------------------------------------"""

    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution) #(256,384,3)
            H, W, C = img.shape 

            detected_map = self.apply_canny(img, low_threshold, high_threshold) #(256,384)
            detected_map = HWC3(detected_map) #(256,384,3)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0 #(256,384,3)
            control = torch.stack([control for _ in range(num_samples)], dim=0) #(1,256,384,3)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone() #(1,3,256,384)

            if seed == -1: #这里是随机种子
                seed = random.randint(0, 65535) 
            seed_everything(seed) #这里是设置随机种子

            if config.save_memory: #这里是设置是否保存内存
                self.model.low_vram_shift(is_diffusing=False) 
            
            """这里是重灾区"""
            cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]} #设置条件参数
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]} #这里用到encode来生成了？
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples, #dim_steps（DDIM算法的步数）、num_samples（生成的样本数量）、shape（样本的形状）、cond（条件）
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond) 
            """samples ([1,4,32,48])"""

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples) #(1,3,256,384)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            #(1,256,384,3)
            results = [x_samples[i] for i in range(num_samples)]
        return results
    
if __name__ == "__main__":
    hk = hackathon()
    hk.initialize()
    path = "/home/player/pictures_croped/bird_"+ "1" + ".jpg"
    img = cv2.imread(path)
    start = datetime.datetime.now().timestamp()
    new_img = hk.process(
            img,
            "a bird", 
            "best quality, extremely detailed", 
            "longbody, lowres, bad anatomy, bad hands, missing fingers", 
            1, #
            256, 
            20,
            False, 
            1, 
            9, 
            2946901, 
            0.0, 
            100, 
            200)
    end = datetime.datetime.now().timestamp()
    print("time cost is: ", (end-start)*1000)
    new_path = "./bird_"+ "1" + ".jpg"
    cv2.imwrite(new_path, new_img[0])