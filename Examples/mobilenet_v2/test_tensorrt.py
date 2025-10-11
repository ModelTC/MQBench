import pycuda.autoinit
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import time
import torch
from common.configs import get_cfg_defaults
from PIL import Image
from calibrator import Preprocess
from dataset import get_dataset
cfg = get_cfg_defaults()


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """Within this context, host_mom means the cpu memory and device means the GPU memory"""
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        shape = engine.get_tensor_shape(binding)
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        size = trt.volume(shape)
        
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    
    # Synchronize the stream
    stream.synchronize()
    
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs


def deserializing_engine(engine_file):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_file, "rb") as f:
        serialized_engine = f.read()
    return runtime.deserialize_cuda_engine(serialized_engine)


def main(engine_file, mode="int8"):
    _, val_dataset, _ = get_dataset()
    val_loaders = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8)

    engine = deserializing_engine(engine_file)

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Do inference
    shape_of_output = (1, 200)
    # Load data to the buffer
    running_corrects = 0.0
    for i, (inps, labels) in enumerate(val_loaders):
        inps = inps.numpy()
        inputs[0].host = inps.reshape(-1)
        t1 = time.time()
        trt_outputs = do_inference(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )  # numpy data
        t2 = time.time()
        feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

        feat = torch.tensor(feat)
        _, preds = torch.max(feat, 1)

        running_corrects += torch.sum(preds == labels.data)

    print(f"Accuracy with TRT {mode} infer : {running_corrects / len(val_dataset) * 100}%")


if __name__ == "__main__":
    # main("fp16")
    engine_file = f"{cfg.SYSTEM.MODELS_DIR}/mbv2_tiny_imagenet_fixed_mse_fbn_v2_best_deploy_model_true.engine"
    main(engine_file, "int8")

# pytorch
# Accuracy : 67.259%

# trt fp16
# Accuracy with TRT fp16 infer : 67.27%

# trt KL INT8
# Accuracy with TRT int8 infer : 66.46%

# qat trt int8
# Accuracy with TRT int8 infer : 70.1%

# qat trt int8 设置 conv_fused为true的话
# Accuracy with TRT int8 infer  69.96%

#####
# dipoorlet MSE INT8
# Accuracy with TRT int8 infer : 66.54000091552734%

# dipoorlet MSE+Brecq INT8
# Accuracy with TRT int8 infer : 67.27999877929688%