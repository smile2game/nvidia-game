from share import *
import torch
from cldm.model import create_model, load_state_dict
import numpy as np
from colored import stylize, fg
import onnx
from polygraphy.backend.common import SaveBytes
from polygraphy.backend.onnx import onnx_from_path


def gen_onnx():
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
    model = model.cuda()
    H = 256
    W = 384

    """----------------------------------------------转换cond_stage_model为engine-----------------"""
    # cond_stage_model = model.cond_stage_model
    # clip = cond_stage_model.transformer #
    # input_ids = torch.zeros((1,77),dtype=torch.int32).to("cuda")
    # input_names = ["input_ids"]
    # output_names = ["context","pooled_output"]
    # print("开始转换clip为onnx")
    # torch.onnx.export(clip,
    #                 (input_ids),
    #                 "sd_clip.onnx",
    #                 export_params=True,
    #                 opset_version=16,
    #                 do_constant_folding=True,
    #                 keep_initializers_as_inputs=False,
    #                 input_names = input_names, 
    #                 output_names = output_names)
    # print("clip转换完成")

    # onnx_model = onnx_from_path("./sd_clip.onnx")
    # # change onnx -inf to -1e4
    # for node in onnx_model.graph.node:
    #     if node.op_type == "ConstantOfShape":
    #         print(node)
    #         attr = node.attribute[0]
    #         print(attr)
    #         if attr.name == "value" and attr.t.data_type == onnx.TensorProto.FLOAT:
    #             np_array = np.frombuffer(attr.t.raw_data, dtype=np.float32).copy()
    #             print("raw array", np_array)
    #             np_array[np_array == -np.inf] = -100000  # 将所有负无穷的值改为-100000
    #             attr.t.raw_data = np_array.tobytes() 
    #             print("new array", np_array)
    #         print(attr)
    # onnx.save_model(
    #     onnx_model,
    #     "sd_clip.onnx",
    # )

    """-----------------------------------------------转换control_model为engine-----------------------------------------------"""
    control_model = model.control_model
    x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
    h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
    t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
    c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")
    output_names = []
    for i in range(13):
        output_names.append("out_"+ str(i))

    torch.onnx.export(control_model,               
                        (x_in, h_in, t_in, c_in),  
                        "sd_control.onnx",   
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        keep_initializers_as_inputs=False,
                        input_names = ['x_in', "h_in", "t_in", "c_in"], 
                        output_names = output_names)
    print("controlnet成功转为engine模型")

    """-----------------------------------------------转换diffusion_model为engine-----------------------------------------------"""
    diffusion_model = model.model.diffusion_model #找对了
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
                        "sd_diffusion.onnx",   
                        export_params=True,
                        opset_version=17,
                        keep_initializers_as_inputs=False,
                        do_constant_folding=True,
                        input_names =input_names, 
                        output_names = output_names)
    print("转换diffusion_model为onnx成功！")


if __name__ == "__main__":
    gen_onnx()