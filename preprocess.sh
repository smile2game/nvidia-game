echo "preprocess"
python3 ./trans_onn.py
trtexec --onnx=./sd_control_fp16-test.onnx --saveEngine=./sd_control_fp16-test.engine --fp16 --verbose=True
trtexec --onnx=./sd_diffusion_fp16-test.onnx --saveEngine=./sd_diffusion_fp16-test.engine --fp16 --verbose=True
# python3 ./trans_engine_local.py

