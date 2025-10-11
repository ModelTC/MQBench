import tensorrt as trt
import os
from common.configs import get_cfg_defaults
from calibrator import Calibrator, CalibDataLoader

cfg = get_cfg_defaults()
LOGGER = trt.Logger(trt.Logger.VERBOSE)


def buildEngine(
    onnx_file, engine_file, mode, data_loader, calibration_table_path
):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    config = builder.create_builder_config()
    parser.parse_from_file(onnx_file)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 * (1 << 20))

    if mode == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    elif mode == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Calibrator(data_loader, calibration_table_path)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("EXPORT ENGINE FAILED!")

    with open(engine_file, "wb") as f:
        f.write(engine)


def main(mode="int8"):
    # onnx_file = f"{cfg.SYSTEM.MODELS_DIR}/mobile_v2_best_model_basic_tiny.onnx"
    engine_file = f"{cfg.MQBENCH.mqbench_log_dir}/mobilev2_model_{mode}_tiny.engine"
    calibration_cache = f"{cfg.MQBENCH.mqbench_log_dir}/trt/mobilev2_model_calib.cache"
    
    dataloader = CalibDataLoader(batch_size=1, calib_count=1000)

    if not os.path.exists(onnx_file):
        print("LOAD ONNX FILE FAILED: ", onnx_file)

    print("Load ONNX file from:%s \nStart export, Please wait a moment..." % (onnx_file))
    buildEngine(onnx_file, engine_file, mode, dataloader, calibration_cache)
    print("Export ENGINE success, Save as: ", engine_file)


if __name__ == "__main__":
    mode = "int8"
    # mode = "fp16"
    main(mode)