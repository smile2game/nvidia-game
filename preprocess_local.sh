echo "preprocess"
python3 ./trans_onn_local.py
trtexec --onnx=./models/onnxmodels/sd_control_fp16-test.onnx --saveEngine=./models/enginemodels/sd_control_fp16-test.engine --fp16 --verbose=True
trtexec --onnx=./models/onnxmodels/sd_diffusion_fp16-test.onnx --saveEngine=./models/enginemodels/sd_diffusion_fp16-test.engine --fp16 --verbose=True


