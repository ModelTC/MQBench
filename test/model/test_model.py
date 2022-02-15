import os
import torch
import unittest

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.utils.logger import logger


class TestQuantizeModel(unittest.TestCase):
    def test_model_ppl(self):
        exclude_list = ['googlenet', 'deeplabv3_mobilenet_v3_large', 'inception_v3', 'lraspp_mobilenet_v3_large',
                        'mobilenet_v3_large', 'mobilenet_v3_small']
        entrypoints = torch.hub.list('pytorch/vision:v0.11.0', force_reload=False)
        for entrypoint in entrypoints:
            if entrypoint in exclude_list:
                continue
            logger.info(f'testing {entrypoint}')
            if 'deeplab' in entrypoint or 'fcn' in entrypoint:
                model_to_quantize = torch.hub.load('pytorch/vision:v0.11.0', entrypoint, pretrained=False, pretrained_backbone=False)
            else:
                model_to_quantize = torch.hub.load('pytorch/vision:v0.11.0', entrypoint, pretrained=False)
            dummy_input = torch.randn(8, 3, 224, 224, device='cpu')
            model_to_quantize.train()
            model_prepared = prepare_by_platform(model_to_quantize, BackendType.PPLW8A16)
            enable_calibration(model_prepared)
            model_prepared(dummy_input)
            enable_quantization(model_prepared)
            output = model_prepared(dummy_input)
            try:
                loss = output.sum()
            except AttributeError:
                loss = output['out'].sum()
            loss.backward()
            model_prepared.eval()
            convert_deploy(model_prepared, BackendType.PPLW8A16, {'x': [1, 3, 224, 224]}, model_name='{}.onnx'.format(entrypoint))
            os.remove('{}.onnx'.format(entrypoint))