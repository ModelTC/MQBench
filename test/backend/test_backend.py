import torch
import unittest

from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization


class TestQuantizeBackend(unittest.TestCase):
    def test_quantize_acedemic(self):
        model_to_quantize = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'FixedFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
            'w_qscheme': {
                'bit': 4,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False
            },
            'a_qscheme': {
                'bit': 4,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False
            }
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        model_prepared = prepare_qat_fx_by_platform(model_to_quantize, BackendType.Academic, prepare_custom_config_dict)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Academic, {'x': [1, 3, 224, 224]}, model_name='resnet18_acedemic_4bit.onnx')

    def test_quantize_tensorrt(self):
        model_to_quantize = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        model_prepared = prepare_qat_fx_by_platform(model_to_quantize, BackendType.Tensorrt)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='resnet18_trt.onnx')

    def test_quantize_nnie(self):
        model_to_quantize = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        model_prepared = prepare_qat_fx_by_platform(model_to_quantize, BackendType.NNIE)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.NNIE, {'x': [1, 3, 224, 224]}, model_name='resnet18_nnie.onnx')

    def test_quantize_snpe(self):
        model_to_quantize = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        model_prepared = prepare_qat_fx_by_platform(model_to_quantize, BackendType.SNPE)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.SNPE, {'x': [1, 3, 224, 224]}, model_name='resnet18_snpe.onnx')

    def test_quantize_pplw8a16(self):
        model_to_quantize = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        dummy_input = torch.randn(2, 3, 224, 224, device='cpu')
        model_to_quantize.train()
        model_prepared = prepare_qat_fx_by_platform(model_to_quantize, BackendType.PPLW8A16)
        enable_calibration(model_prepared)
        model_prepared(dummy_input)
        enable_quantization(model_prepared)
        loss = model_prepared(dummy_input).sum()
        loss.backward()
        model_prepared.eval()
        convert_deploy(model_prepared, BackendType.PPLW8A16, {'x': [1, 3, 224, 224]}, model_name='resnet18_pplw8a16.onnx')