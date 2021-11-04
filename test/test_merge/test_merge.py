import torch
import unittest

from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_merge_bn
from mqbench.utils.state import enable_calibration, enable_quantization, disable_all


class TestMergeBN(unittest.TestCase):

    def test_case_1(self):

        def cos(a, b):
            return (a * b).sum() / torch.sqrt((a ** 2).sum()) / torch.sqrt((b ** 2).sum())

        dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
        extra_qconfig_dict = {
            'w_observer': 'MinMaxObserver',
            'a_observer': 'EMAMinMaxObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
        prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
        # First model
        model_1 = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=False)
        model_1 = prepare_by_platform(model_1, BackendType.Tensorrt, prepare_custom_config_dict)
        model_1.train()
        enable_calibration(model_1)
        model_1(dummy_input)
        enable_quantization(model_1)
        model_1.eval()
        prev_output = model_1(dummy_input)
        # Convert model
        convert_merge_bn(model_1)
        new_output = model_1(dummy_input)
        # Test
        # merge bn import about 1e-8 mean error.
        self.assertTrue(cos(new_output, prev_output) >= 0.999)