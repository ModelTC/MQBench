import numpy as np
import argparse
from ptq.data.imagenet import load_data
from ptq.models import load_model
from ptq.utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform, BackendType


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data


def get_quantize_model(model, config):
    backend_dict = {
            'Academic': BackendType.Academic,
            'Tensorrt': BackendType.Tensorrt,
            'SNPE': BackendType.SNPE,
            'PPLW8A16': BackendType.PPLW8A16,
            'NNIE': BackendType.NNIE,
            'Vitis': BackendType.Vitis,
            'ONNX_QNN': BackendType.ONNX_QNN,
            'PPLCUDA': BackendType.PPLCUDA,
    }
    backend_type = BackendType.Academic if not hasattr(
        config, 'backend') else backend_dict[config.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, extra_prepare_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Solver')
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()
    config = parse_config(args.config)
    # seed first
    seed_all(config.process.seed)
    # load_model
    model = load_model(config.model)
    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
    model.cuda()
    # load_data
    train_loader, val_loader = load_data(**config.data)
    # evaluate
    if not hasattr(config, 'quantize'):
        evaluate(val_loader, model)
    else:
        assert config.quantize.quantize_type == 'ptq'
        print('begin calibration now!')
        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].cuda())
        print('begin quantization now!')
        enable_quantization(model)
        evaluate(val_loader, model)

