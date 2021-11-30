import copy
import operator
from collections import OrderedDict
from typing import (
    Dict, Any, Callable, NoReturn
)

import torch
import torch.nn.intrinsic as nni
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
from torch.quantization.quantize_fx import (
    _fuse_fx
)

import mqbench
import mqbench.nn as qnn
import mqbench.nn.intrinsic as qnni 
import mqbench.nn.intrinsic.qat as qnniqat
from mqbench.utils.logger import logger
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.fake_quantize.tqt import TqtFakeQuantize


@register_model_quantizer(BackendType.NNIE)
class ModelQuantizer(object):
    """General model quantizer class.
    First, replace common float module to nn.qat.modules to make weight fake
    quantized.
    Second, insert activation fake quantize node before specific layers. Layer
    type is defined in function_type_to_quant_input / module_type_to_quant_input.
    We only quantize the inputs of layers and leave the output not quantized
    since it is next layer's input.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        self.additional_function_type = extra_quantizer_dict.get('additional_function_type', [])
        self.additional_module_type = extra_quantizer_dict.get('additional_module_type', ())
        self.additional_fuser_method_mapping = extra_fuse_dict.get('additional_fuser_method_mapping', {})
        self.additional_fusion_pattern = extra_fuse_dict.get('additional_fusion_pattern', {})
        self.additional_qat_module_mapping = extra_fuse_dict.get('additional_qat_module_mapping', {})
        self.exclude_module_name = extra_quantizer_dict.get('exclude_module_name', [])
        self.exclude_function_type = extra_quantizer_dict.get('exclude_function_type', [])
        self.exclude_node_name = extra_quantizer_dict.get('exclude_node_name', [])
        self.extra_fuse_dict = extra_fuse_dict

    def prepare(self, model: GraphModule, qconfig):
        model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        return model

    def _insert_fake_quantize_for_act_quant(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()

        for node in node_to_quantize_output:
            fake_quantizer = qconfig.activation()
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

    def _fix_succ_recursivly(self, args, target_node, inserted_node):
        # List / Tuple
        if isinstance(args, (list, tuple)):
            _tmp = list(args)
            for _i, _arg in enumerate(args):
                if _arg == target_node:
                    _tmp[_i] = inserted_node
                elif isinstance(_arg, tuple):
                    _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
                elif isinstance(_arg, list):
                    _tmp[_i] = list(self._fix_succ_recursivly(_arg, target_node, inserted_node))
                elif isinstance(_arg, dict):
                    _tmp[_i] = self._fix_succ_recursivly(_arg, target_node, inserted_node)
            return tuple(_tmp)
        # Dict
        elif isinstance(args, dict):
            _tmp = {}
            for k, v in args.items():
                if v == target_node:
                    _tmp[k] = inserted_node
                elif not isinstance(v, torch.fx.node.Node):
                    _tmp[k] = self._fix_succ_recursivly(v, target_node, inserted_node)
                else:
                    _tmp[k] = v
            return _tmp
        else:
            raise NotImplementedError('{} can not be handled now.'.format(type(args_tuple)))

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model

    @property
    def implicit_merge_patterns(self) -> list:
        # Layers which do not need quantize among them.
        # In reversed order!
        return [
            (operator.add, operator.mul)
        ]

    def _on_merge_chain(self, modules, pattern, pair, p_pos=0, v_pos=0):
        if v_pos == len(pair):
            return True
        if p_pos == len(pattern):
            return v_pos == len(pair)
        node = pair[v_pos]
        cur_pattern = pattern[p_pos]
        # Means current node is matched.
        if (node.op == "call_module" and type(modules[node.target]) == cur_pattern) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target == cur_pattern):
            # Means compairing pair.
            if len(pattern) > p_pos and len(pair) > v_pos:
                return self._on_merge_chain(modules, pattern, pair, p_pos + 1, v_pos + 1)
            # Means compairing extra node.
            matched = False
            flatten_args = self._flatten_args(node.args)
            for _arg in flatten_args:
                extra_pair = (*pair, _arg)
                if isinstance(_arg, torch.fx.node.Node) and \
                        self._on_merge_chain(modules, pattern, extra_pair, p_pos + 1, v_pos + 1):
                    matched = True
            return matched
        # Current node is not matched, skip to next.
        else:
            return self._on_merge_chain(modules, pattern, pair, p_pos + 1, v_pos)

    def _is_implicit_merge(self, modules, pair):
        for pattern in self.implicit_merge_patterns:
            if self._on_merge_chain(modules, pattern, pair):
                return True
        return False

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
            # ConvTranspose
            torch.nn.ConvTranspose2d,
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
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input):
                input_node_list = self._flatten_args(node.args)
                for _node in input_node_list:
                    if isinstance(_node, torch.fx.node.Node):
                        if self._is_implicit_merge(modules, (node, _node)):
                            logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                            continue
                        node_need_to_quantize_output.append(_node)
        return node_need_to_quantize_output

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


@register_model_quantizer(BackendType.Academic)
class AcademicQuantizer(ModelQuantizer):
    """Academic setting mostly do not merge BN and leave the first and last layer to higher bits.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.io_module = {}
        self.post_act_8bit_node_name = []

    def prepare(self, model: GraphModule, qconfig):
        self._get_io_module(model)
        self._get_post_act_8bit_node_name(model)
        model = self._weight_quant(model, qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        return model

    def _weight_quant(self, model: GraphModule, qconfig):
        logger.info("Replace module to qat module.")
        wqconfig_8bit = copy.deepcopy(qconfig)

        wq_symmetry = True if qconfig.weight.p.keywords['qscheme'] == torch.per_channel_symmetric or qconfig.weight.p.keywords[
            'qscheme'] == torch.per_tensor_symmetric else False
        wqconfig_8bit.weight.p.keywords['quant_min'] = -2 ** (8 - 1) if wq_symmetry else 0
        wqconfig_8bit.weight.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if wq_symmetry else 2 ** 8 - 1

        # wqconfig_8bit.weight.p.keywords['quant_min'] = -128
        # wqconfig_8bit.weight.p.keywords['quant_max'] = 127
        for name, module in model.named_modules():
            if name in self.io_module.keys():
                logger.info("Set layer {} to 8 bit.".format(name))
                module.qconfig = wqconfig_8bit
        flattened_qconfig_dict = get_flattened_qconfig_dict({'': qconfig})
        propagate_qconfig_(model, flattened_qconfig_dict)
        self._qat_swap_modules(model, self.additional_qat_module_mapping)
        return model

    @property
    def function_type_to_quant_input(self) -> list:
        return self.additional_function_type

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Conv
            torch.nn.qat.modules.conv.Conv2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
        ) + self.additional_module_type

    def _get_post_act_8bit_node_name(self, model):
        for node in self.io_module.values():
            for _arg in node.args:
                if isinstance(_arg, torch.fx.node.Node):
                    self.post_act_8bit_node_name.append(_arg.name)

    def _get_io_module(self, model):
        total_args = []
        nodes = list(model.graph.nodes)
        for node in nodes:
            the_first_layer = False
            for _arg in node.args:
                if isinstance(_arg, torch.fx.node.Node):
                    if _arg.op == 'placeholder':
                        the_first_layer = True
                    total_args.append(_arg.name)
            if the_first_layer:
                self.io_module[node.target] = node
            if node.op == 'output':
                for _arg in node.args:
                    if isinstance(_arg, torch.fx.node.Node):
                        self.io_module[_arg.target] = _arg

    def _insert_fake_quantize_for_act_quant(self, model: GraphModule, qconfig):
        graph = model.graph
        nodes = list(model.graph.nodes)

        quantizer_prefix = "_post_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()

        aqconfig_8bit = copy.deepcopy(qconfig.activation)

        aq_symmetry = True if qconfig.activation.p.keywords['qscheme'] == torch.per_channel_symmetric or \
            qconfig.activation.p.keywords['qscheme'] == torch.per_tensor_symmetric else False

        aqconfig_8bit.p.keywords['quant_min'] = -2 ** (8 - 1) if aq_symmetry else 0
        aqconfig_8bit.p.keywords['quant_max'] = 2 ** (8 - 1) - 1 if aq_symmetry else 2 ** 8 - 1
        for node in node_to_quantize_output:
            if node.name in self.post_act_8bit_node_name:
                logger.info("Set {} post act quantize to 8 bit.".format(node.name))
                fake_quantizer = aqconfig_8bit()
            else:
                fake_quantizer = qconfig.activation()
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


@register_model_quantizer(BackendType.Tensorrt)
class TRTModelQuantizer(ModelQuantizer):
    """The different points of TRT quantizer are how to deal with add op
    and the last layer.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)

    @property
    def _merge_add_type(self):
        return (torch.nn.Conv2d, torch.nn.Linear)

    def _find_act_quants(self, model: GraphModule) -> set:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        for node in nodes:
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input):
                # Add will be merged with previous conv.
                input_node_list = list(filter(lambda x: isinstance(x, torch.fx.node.Node),
                                              self._flatten_args(node.args)))
                if node.target is operator.add:
                    merge_node = self._find_add_merge_node(model, input_node_list, node)
                    if merge_node:
                        input_node_list.remove(merge_node)
                    node_need_to_quantize_output.extend(input_node_list)
                else:
                    for _node in input_node_list:
                        if self._is_implicit_merge(modules, (node, _node)):
                            continue
                        if isinstance(_node, torch.fx.node.Node):
                            node_need_to_quantize_output.append(_node)
        return node_need_to_quantize_output

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


@register_model_quantizer(BackendType.PPLCUDA)
@register_model_quantizer(BackendType.SNPE)
@register_model_quantizer(BackendType.PPLW8A16)
class TotalINTQuantizer(ModelQuantizer):
    """There is only INT8 calculations in the model.
    We quantize the input tensors and output tensors of all layers,
    except those in _passed_func_type and _passed_module_type.
    For example add + relu pattern, there is no need to insert fake
    quantize node between them.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)

    @property
    def _passed_func_type(self):
        return (
            torch.nn.functional.relu, 
            torch.nn.functional.relu6,
            torch.flatten
        )

    @property
    def _passed_module_type(self):
        return (
            torch.nn.ReLU,
            torch.nn.ReLU6
        )

    def _find_act_quants(self, model: GraphModule) -> list:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = super(). _find_act_quants(model)
        for node in nodes:
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input):
                for next_node in node.users:
                    if not ((next_node.op == 'call_function' and next_node.target in self._passed_func_type) or
                            (next_node.op == 'call_module' and isinstance(modules[next_node.target], self._passed_module_type))):
                        node_need_to_quantize_output.append(node)
                    else:
                        node_need_to_quantize_output.append(next_node)
        return node_need_to_quantize_output


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
            nni.ConvBn2d: qnniqat.ConvBn2d,
            nni.ConvBnReLU2d: qnniqat.ConvBnReLU2d,
            nni.ConvReLU2d: qnniqat.ConvReLU2d,
        }

    @property
    def module_type_to_quant_input(self) -> tuple:
        return super().module_type_to_quant_input + (
            torch.nn.Conv2d,
            qnni.ConvBn2d, 
            qnni.ConvReLU2d,
            qnni.ConvBnReLU2d
        )

    def prepare(self, model: GraphModule, qconfig):
        model = _fuse_fx(model, self.extra_fuse_dict)
        model = self._weight_quant(model, qconfig)
        model = self._insert_fake_quantize_for_act_quant(model, qconfig)
        prepared = model
        self._set_quant_type(prepared)
        return prepared


    def _find_act_quants(self, model: GraphModule) -> (set, set):
        node_need_to_quantize_output = super(). _find_act_quants(model)
        nodes = list(model.graph.nodes)
        for node in nodes:
            if node.op == 'output':
                for output_node in self._flatten_args(node.args):
                    assert isinstance(output_node, torch.fx.node.Node)
                    node_need_to_quantize_output.append(output_node)

        return set(node_need_to_quantize_output)

    def _find_input_quants(self, model) -> (set, set):
        node_need_to_quantize_weight = set() 
        nodes = list(model.graph.nodes)
        for node in nodes:
            if node.op == 'placeholder' and node.all_input_nodes == []:
                node_need_to_quantize_weight.add(list(node.users)[0])
        return set(node_need_to_quantize_weight)


    def _find_weight_quants(self, model) -> (set, set):
        node_need_to_quantize_weight = set() 
        nodes = list(model.graph.nodes)
        module_dict = dict(model.named_modules())
        for node in nodes:
            if node.target in module_dict:
                if hasattr(module_dict[node.target], 'weight_fake_quant') or hasattr(module_dict[node.target], 'bias_fake_quant'):
                    node_need_to_quantize_weight.add(node)
        return set(node_need_to_quantize_weight)

    def _set_quant_type(self, model: GraphModule) -> NoReturn:
        tensor_type_set = self._find_act_quants(model)
        params_type_set = self._find_weight_quants(model)
        inputs_type_set = self._find_input_quants(model)
        module_dict = dict(model.named_modules())

        for node in tensor_type_set:
            if isinstance(node.target, str) and node.target in module_dict:
                next_op = module_dict[node.target]
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


@register_model_quantizer(BackendType.ONNX_QNN)
class TVMQuantizer(ModelQuantizer):
    """Quantize model according to TVM ONNX frontend.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)

    @property
    def _relu_module_type(self):
        return (torch.nn.ReLU, torch.nn.ReLU6)

    @property
    def _relu_function_type(self):
        return (torch.nn.functional.relu, torch.nn.functional.relu6)

    def _find_act_quants(self, model: GraphModule) -> (set, set):
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = super(). _find_act_quants(model)
        for node in nodes:
            if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or \
                    node.name in self.exclude_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input):
                # Add current node if not merge relu.
                for next_node in node.users:
                    if not ((next_node.op == 'call_function' and next_node.target in self._relu_function_type) or (
                            next_node.op == 'call_module' and isinstance(modules[next_node.target], self._relu_module_type))):
                        node_need_to_quantize_output.append(node)
                    else:
                        node_need_to_quantize_output.append(next_node)
        return node_need_to_quantize_output

    def _qat_swap_modules(self, root: GraphModule, additional_qat_module_mapping: Dict[Callable, Callable]):
        all_mappings = get_combined_dict(
            get_default_qat_module_mappings(), additional_qat_module_mapping)
        # There is no QLinearFC in ONNX for now.
        del(all_mappings[torch.nn.modules.linear.Linear])
        del(all_mappings[torch.nn.modules.linear._LinearWithBias])
        del(all_mappings[torch.nn.intrinsic.modules.fused.LinearReLU])
        del(all_mappings[mqbench.nn.intrinsic.modules.fused.LinearBn1d])
        root = self._convert(root, all_mappings, inplace=True)
        return root

    @property
    def function_type_to_quant_input(self) -> list:
        return [
            operator.add,
            # TODO operator.mul,
            # TODO torch.cat,
            torch.nn.functional.adaptive_avg_pool2d
            # sigmoid
            # TODO torch.nn.functional.sigmoid
        ]

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            # Conv
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d,
            torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d,
            # Linear
            torch.nn.qat.modules.linear.Linear,
            qnn.intrinsic.qat.LinearBn1d,
            # Pooling
            torch.nn.modules.pooling.AvgPool2d,
            torch.nn.modules.pooling.AdaptiveAvgPool2d,
            # Prelu
            # TODO torch.nn.PReLU,
        )

    @property
    def implicit_merge_patterns(self) -> list:
        # Layers which do not need quantize among them.
        # In reversed order!
        return [
            (torch.nn.ReLU, operator.add)
        ]
