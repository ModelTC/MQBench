import operator
from typing import (
    Dict, Any, Callable
)

import torch
from torch.fx import (
    GraphModule
)
from torch.quantization import (
    propagate_qconfig_, 
    swap_module
)
from torch.nn.intrinsic import (
    _FusedModule
)
from torch.quantization.quantization_mappings import (
    get_default_qat_module_mappings,
    get_default_static_quant_module_mappings
)
from torch.quantization.utils import (
    get_combined_dict
)
from torch.quantization.fx.qconfig_utils import (
    get_flattened_qconfig_dict
)

import mqbench.nn as qnn
import mqbench.nn.intrinsic
import mqbench.nn.intrinsic.qat # noqa F401
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType


@register_model_quantizer(BackendType.SNPE)
@register_model_quantizer(BackendType.NNIE)
@register_model_quantizer(BackendType.Academic)
class ModelQuantizer(object):
    """General model quantizer class.
    First, replace common float module to nn.qat.modules to make weight fake
    quantized.
    Second, insert activation fake quantize node before specific layers. Layer
    type is defined in function_type_to_quant_input / module_type_to_quant_input.
    We leave the output not quantized since it is next layer's input.
    """
    def __init__(self, extra_quantizer_dict):
        self.additional_function_type = extra_quantizer_dict.get('additional_function_type', [])
        self.additional_module_type = extra_quantizer_dict.get('additional_module_type', ())
        self.exclude_module_name = extra_quantizer_dict.get('exclude_module_name', [])

    def prepare(self, model: GraphModule, qconfig_dict: Dict):
        model = self._weight_quant(model, qconfig_dict)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig_dict)
        return model

    def _insert_fake_quantize_for_act_quant(
            self, 
            model: GraphModule, 
            qconfig_dict: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)

        for node in node_to_quantize_output:
            fake_quantizer = qconfig_dict.activation()
            quantizer_name = node.name + quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        model.recompile()
        model.graph.lint()
        return model

    def _fix_succ_recursivly(self, args_tuple, target_node, inserted_node):
        _tmp = list(args_tuple)
        for _i, _arg in enumerate(args_tuple):
            if _arg == target_node:
                _tmp[_i] = inserted_node
            elif isinstance(_arg, tuple):
                _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
            elif isinstance(_arg, list):
                _tmp[_i] = list(self._fix_succ_recursivly(_arg, target_node, inserted_node))
        args_tuple = tuple(_tmp)
        return args_tuple

    def _weight_quant(self, model: GraphModule, qconfig_dict: Dict):
        logger.info("Replace module to qat module.")
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig_dict})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self._additional_qat_module_mapping)
        return model

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            operator.add, 
            operator.mul,
            torch.cat,
            torch.nn.functional.adaptive_avg_pool2d
        ] + self.additional_function_type

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Conv
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            torch.nn.qat.modules.conv.Conv2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
            qnn.intrinsic.qat.LinearBn1d,
            # Pooling
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.pooling.AvgPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # BN
            torch.nn.BatchNorm2d,
            # Prelu mostly do not merge.
            torch.nn.PReLU,
            # Upsample
            torch.nn.Upsample
        ) + self.additional_module_type

    def _flatten_args(self, node):
        flattned_args = []
        if isinstance(node, dict):
            for v in node.values():
                flattned_args.extend(self._flatten_args(v))
        elif isinstance(node, tuple) or isinstance(node, list):
            for n in node:
                flattned_args.extend(self._flatten_args(n))
        else:
            flattned_args.extend([node])
        return flattned_args

    def _find_act_quants(self, model: GraphModule) -> (set, set):
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        for node in nodes:
            if node.op == "call_module" and node.target in self.exclude_module_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and 
                    node.target in self.function_type_to_quant_input):
                input_node_list = self._flatten_args(node.args)
                for _node in input_node_list:
                    if isinstance(_node, torch.fx.node.Node):
                        node_need_to_quantize_output.append(_node)
        return set(node_need_to_quantize_output)

    @property
    def _additional_qat_module_mapping(self):
        return {
            qnn.intrinsic.LinearBn1d: qnn.intrinsic.qat.LinearBn1d
        }

    def _qat_swap_modules(self, root: GraphModule, additional_qat_module_mapping: Dict[Callable, Callable]):
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        root = self._convert(root, all_mappings, inplace=True)
        return root

    def _convert(self, module, mapping=None, inplace=False, scope=''):
        if mapping is None:
            mapping = get_default_static_quant_module_mappings()

        if not inplace:
            module = copy.deepcopy(module)
        reassign = {}
        for name, mod in module.named_children():
            # fused modules are swapped as one unit
            new_scope = "{}.{}".format(scope, name) if scope != '' else name
            if new_scope in self.exclude_module_name:
                logger.info("Skip quant layer: " + new_scope)
                continue
            if not isinstance(mod, _FusedModule):
                self._convert(mod, mapping, True, new_scope)
            reassign[name] = swap_module(mod, mapping, {})

        for key, value in reassign.items():
            module._modules[key] = value

        return module


@register_model_quantizer(BackendType.Tensorrt)
class TRTModelQuantizer(ModelQuantizer):
    """The different points of TRT quantizer are how to deal with add op
    and the last layer.
    """
    def __init__(self, extra_quantizer_dict):
        super().__init__(extra_quantizer_dict)

    @property
    def _merge_add_type(self):
        return (torch.nn.Conv2d, torch.nn.Linear)

    def _find_act_quants(self, model: GraphModule) -> set:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        for node in nodes:
            if node.op == "call_module" and node.target in self.exclude_module_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and 
                    node.target in self.function_type_to_quant_input):
                # Add will be merged with previous conv.
                input_node_list = self._flatten_args(node.args)
                if node.target is operator.add:
                    merge_node = self._find_add_merge_node(model, input_node_list, node)
                    if merge_node:
                        input_node_list.remove(merge_node)
                    node_need_to_quantize_output.extend(input_node_list)
                else:
                    for _node in input_node_list:
                        if isinstance(_node, torch.fx.node.Node):
                            node_need_to_quantize_output.append(_node)

        return set(node_need_to_quantize_output)

    def _find_add_merge_node(self, model, input_node_list, node):
        """Find the first input node which has only one successor from the last.
        This kind of node can be merge with add.
        """
        input_node_list.reverse()
        modules = dict(model.named_modules())
        for input_node in input_node_list:
            if input_node.op == 'call_module' and type(modules[input_node.target]) in self._merge_add_type:
                succ = 0
                for _node in list(model.graph.nodes):
                    _node_input_list = self._flatten_args(_node.args)
                    if input_node in _node_input_list:
                        succ += 1
                if succ == 1:
                    return input_node
        return None


@register_model_quantizer(BackendType.PPLW8A16)
class TotalINTQuantizer(ModelQuantizer):
    """There is only INT8 calculations in the model.
    We quantize the input tensors of all layers and the output tensors
    of the last layers. We quantize every activations tensors and weight
    tensors using this method.
    """
    def __init__(self, extra_quantizer_dict):
        super().__init__(extra_quantizer_dict)

    def _find_act_quants(self, model: GraphModule) -> (set, set):
        node_need_to_quantize_output = super(). _find_act_quants(model)
        nodes = list(model.graph.nodes)
        for node in nodes:
            if node.op == 'output':
                for output_node in self._flatten_args(node.args):
                    node_need_to_quantize_output.add(output_node)

        return set(node_need_to_quantize_output)