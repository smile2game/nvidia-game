echo "preprocess"
python3 ./trans_onn_local.py
trtexec --onnx=./models/onnxmodels/sd_control_fp16-test.onnx --saveEngine=./models/enginemodels/sd_control_fp16-test.engine --fp16 --verbose=True 
trtexec --onnx=./models/onnxmodels2/sd_diffusion_fp16-test.onnx --saveEngine=./models/enginemodels/sd_diffusion_fp16-test-l5.engine --fp16 --verbose=True --builderOptimizationLevel=5
trtexec --onnx=./models/onnxmodels/sd_clip_fp16-test-2211.onnx --saveEngine=./models/enginemodels/sd_clip_fp16-test-1735.engine --verbose=True --fp16 


