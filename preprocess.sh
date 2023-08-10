echo "preprocess"
# python3 trans_onn.py
# # trtexec --onnx=sd_clip_fp32.onnx --saveEngine=sd_clip_fp32.plan  --verbose=True --builderOptimizationLevel=5 
# trtexec --onnx=sd_control.onnx --saveEngine=sd_control_fp16.engine --fp16 --verbose=True --builderOptimizationLevel=5
# trtexec --onnx=sd_diffusion.onnx --saveEngine=sd_diffusion_fp16.engine --fp16 --verbose=True --builderOptimizationLevel=5
# trtexec --onnx=./sd_control_test.onnx --saveEngine=./sd_control_fp16.engine --fp16 --verbose
# trtexec --onnx=./sd_diffusion_test.onnx --saveEngine=./sd_diffusion_fp16.engine --fp16 --verbose
# # trtexec --onnx=./sd_vae.onnx --saveEngine=./sd_vae_fp16.engine --fp16 --optShapes=z:1x4x32x48 
# trtexec --onnx=./sd_vae.onnx --saveEngine=./sd_vae_fp16-b5.engine --fp16 --optShapes=z:1x4x32x48  --builderOptimizationLevel=5
