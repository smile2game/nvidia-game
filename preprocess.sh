echo "preprocess"
python3 ./trans_onn.py
trtexec --onnx=./sd_control_fp16.onnx --saveEngine=./sd_control_fp16.engine --fp16 --verbose=True 
trtexec --onnx=./sd_diffusion_fp16.onnx --saveEngine=./sd_diffusion_fp16.engine --fp16 --verbose=True 
# trtexec --onnx=./sd_control_fp16.onnx --saveEngine=./sd_control_fp16.engine --fp16 --verbose=True --builderOptimizationLevel=5
# trtexec --onnx=./sd_diffusion_fp16.onnx --saveEngine=./sd_diffusion_fp16.engine --fp16 --verbose=True --builderOptimizationLevel=5
# python3 ./trans_engine_local.py

