'''
version: 1.0.0
Author: BruceCui
Date: 2024-11-13 16:57:30
LastEditors: BruceCui
LastEditTime: 2024-12-03 19:32:57
Description: 根据trt量化参数, 生成新的 tensorrt engine
'''
import tensorrt as trt
import os, sys
import json
from spectrautils.print_utils import *
from common.configs import get_cfg_defaults

cfg = get_cfg_defaults()
LOGGER = trt.Logger(trt.Logger.VERBOSE)


def set_dynamic_range(config, network, blob_range):
    config.flags |= 1 << int(trt.BuilderFlag.INT8)
    # TODO: does STRICT_TYPES flag really needed?
    # config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)
    # config.int8_calibrator = None
    for layer in network:
        if layer.type != trt.LayerType.SHAPE and \
            layer.type != trt.LayerType.CONSTANT and \
            layer.type != trt.LayerType.CONCATENATION and \
            layer.type != trt.LayerType.GATHER:
            layer.precision = trt.DataType.INT8

        for i in range(layer.num_inputs):
            inp = layer.get_input(i)
            if inp is not None and inp.name in blob_range:
                dmax = blob_range[inp.name]
                if inp.dynamic_range is None:
                    inp.set_dynamic_range(-dmax, dmax)
                    print(f'set dynamic range of tensor "{inp.name}" to {dmax}.')

        for i in range(layer.num_outputs):
            output = layer.get_output(i)
            if output.name in blob_range:
                dmax = blob_range[output.name]
                if output.dynamic_range is None:
                    output.set_dynamic_range(-dmax, dmax)
                    print(f'set dynamic range of tensor "{output.name}" to {dmax}.')

def buildEngine(onnx_file, export_engine_file, json_path):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    with open(json_path, 'r') as f:
        dipoorlet_range = json.load(f)
    
    set_dynamic_range(config, network, dipoorlet_range["tensorrt"]["blob_range"])
    
    config.int8_calibrator = None
    engine = builder.build_serialized_network(network, config)
    
    if engine is None:
        print("EXPORT ENGINE FAILED!")

    with open(export_engine_file, "wb") as f:
        f.write(engine)


def main(onnx_file, json_path, export_engine_file):
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)
    print("Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file))
    buildEngine(onnx_file, export_engine_file, json_path)
    print_colored_text(f"Export ENGINE success, Save as: \n{export_engine_file}", "green")


if __name__ == "__main__":
    onnx_file = f"{cfg.SYSTEM.MODELS_DIR}/mbv2_tiny_imagenet_fixed_mse_fbn_v2_best_true_deploy_model.onnx"
    json_path = f"{cfg.SYSTEM.MODELS_DIR}/mbv2_tiny_imagenet_fixed_mse_fbn_v2_best_true_clip_ranges.json"
    export_engine_file = f"{cfg.SYSTEM.MODELS_DIR}/mbv2_tiny_imagenet_fixed_mse_fbn_v2_best_deploy_model_true.engine"
    
    main(onnx_file, json_path, export_engine_file)
