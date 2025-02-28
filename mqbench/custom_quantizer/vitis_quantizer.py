import operator
from typing import List, NoReturn

import torch
import torch.nn.intrinsic as nni
from torch.fx import GraphModule
from torch.quantization.quantize_fx import _fuse_fx

import mqbench.nn.intrinsic as qnni 
import mqbench.nn.intrinsic.qat as qnniqat
from mqbench.utils import getitem2node
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.fake_quantize.tqt import TqtFakeQuantize
from mqbench.custom_quantizer.model_quantizer import ModelQuantizer


@register_model_quantizer(BackendType.Vitis)
class VitisQuantizer(ModelQuantizer):
    """There is only INT8 calculations in the model.
    We quantize the input tensors of all layers and the output tensors
    of the last layers. We quantize every activations tensors and weight
    tensors using this method. NOTE: the acti and weight have different 
    quantize type.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.additional_qat_module_mapping = {
            # Intrinsic modules:
            # nni.ConvBn2d: qnniqat.ConvBn2d,
            # nni.ConvBnReLU2d: qnniqat.ConvBnReLU2d,
            # nni.ConvReLU2d: qnniqat.ConvReLU2d,
        }

    @property
    def module_type_to_quant_input(self) -> tuple:
        return super().module_type_to_quant_input + (
            torch.nn.Conv2d,
            qnni.ConvBn2d, 
            qnni.ConvReLU2d,
            qnni.ConvBnReLU2d
        )

    @property
    def module_type_to_quant_output(self) -> tuple:
        return (
            # Conv
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            torch.nn.qat.modules.conv.Conv2d,
            qnniqat.ConvBnReLU2d,
            qnniqat.ConvBn2d,
            qnniqat.ConvReLU2d,
            # ConvTranspose
            torch.nn.ConvTranspose2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
            # Pooling
            torch.nn.modules.pooling.AvgPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # BN
            torch.nn.BatchNorm2d,
            torch.nn.ReLU,
            # Prelu mostly do not merge.
            torch.nn.PReLU,
            torch.nn.Upsample,
        ) 


    @property
    def function_type_to_quant_output(self) -> List:
        return [
            operator.add,
            operator.mul,
            torch.cat,
            torch.nn.functional.adaptive_avg_pool2d,
            torch.nn.functional.avg_pool2d,
            torch.nn.functional.relu,
            torch.nn.functional.conv2d,
            torch.nn.functional.linear,
            torch.nn.functional.interpolate,
        ] 

    def prepare(self, model: GraphModule, qconfig, is_qat, backend_config, freeze_bn):
        model = _fuse_fx(model, is_qat, self.extra_fuse_dict, backend_config)
        model = self._weight_quant(model, qconfig, backend_config, freeze_bn)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        prepared = model
        self._set_quant_type(prepared)
        return prepared


    def _find_act_quants(self, model: GraphModule) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        if hasattr(self, 'node_need_to_quantize_output'):
            return self.node_need_to_quantize_output
        self.node_need_to_quantize_output = []
        g2node = getitem2node(model)
        for node in nodes:
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_output)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_output):
                self.node_need_to_quantize_output.append(node)
                logger.info(f'Add {node.name} to output quantize')
        return self.node_need_to_quantize_output

    def _find_input_quants(self, model) -> List:
        node_need_to_quantize_weight = []
        nodes = list(model.graph.nodes)
        for node in nodes:
            if node.op == 'placeholder' and node.all_input_nodes == []:
                node_need_to_quantize_weight.append(list(node.users)[0])
        return node_need_to_quantize_weight

    def _find_weight_quants(self, model) -> List:
        node_need_to_quantize_weight = []
        nodes = list(model.graph.nodes)
        module_dict = dict(model.named_modules())
        for node in nodes:
            if node.target in module_dict:
                if hasattr(module_dict[node.target], 'weight_fake_quant') or hasattr(module_dict[node.target], 'bias_fake_quant'):
                    node_need_to_quantize_weight.append(node)
        return node_need_to_quantize_weight

    def _set_quant_type(self, model: GraphModule) -> NoReturn:
        tensor_type_set = self._find_act_quants(model)
        params_type_set = self._find_weight_quants(model)
        inputs_type_set = self._find_input_quants(model)
        module_dict = dict(model.named_modules())
        quantizer_prefix = "_post_act_fake_quantizer"

        for node in tensor_type_set:
            if isinstance(node.name, str) and (node.name + quantizer_prefix) in module_dict:
                next_op = module_dict[node.name + quantizer_prefix]
                if isinstance(next_op, TqtFakeQuantize):
                    next_op.set_quant_type('tensor')
                    logger.info(f'{node.target} has been set to quant type <tensor>')
        for node in params_type_set:
            if isinstance(node.target, str) and node.target in module_dict:
                op = module_dict[node.target]
                if hasattr(op, 'weight_fake_quant'):
                    if isinstance(op.weight_fake_quant, TqtFakeQuantize):
                        op.weight_fake_quant.set_quant_type('param')
                        logger.info(f'{node.target} has been set to quant type <param/weight>')
                if hasattr(op, 'bias_fake_quant'):
                    if isinstance(op.bias_fake_quant, TqtFakeQuantize):
                        op.bias_fake_quant.set_quant_type('param')
                        logger.info(f'{node.target} has been set to quant type <param/bias>')
        for node in inputs_type_set:
            if isinstance(node.target, str) and node.target in module_dict:
                next_op = module_dict[node.target]    
                if isinstance(next_op, TqtFakeQuantize):
                    next_op.set_quant_type('input')
                    logger.info(f'{node.target} has been set to quant type <input>')
