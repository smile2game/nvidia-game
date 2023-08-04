import onnx
import numpy as np
import onnxruntime as rt
import cv2
import torch

# model_path = './models/onnxmodels/sd_clip_fp16-test-int32-torch_in.onnx'
model_path = './models/onnxmodels/sd_control_fp16-test.onnx'
# 验证模型合法性
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
H= 256
W= 384
# 读入图像并调整为输入维度
# input_ids = torch.zeros((1,77),dtype = torch.int32)
# input_ids = input_ids1.numpy()

x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")

# 设置模型session以及输入信息
sess = rt.InferenceSession(model_path)
input_name1 = sess.get_inputs()[0].name
input_name2 = sess.get_inputs()[1].name
input_name3 = sess.get_inputs()[2].name
input_name4 = sess.get_inputs()[3].name

output = sess.run(None, {input_name1: x_in,input_name2: h_in,input_name3: t_in,input_name4: c_in})
# output = sess.run(None, {input_name1: input_ids})
print(output)