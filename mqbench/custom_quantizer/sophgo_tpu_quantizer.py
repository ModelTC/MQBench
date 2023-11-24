import operator
import torch
from torch.fx import GraphModule
import torch.nn.intrinsic as nni
import mqbench.nn.intrinsic.qat as qnniqat
import mqbench.nn.intrinsic as qnni
from mqbench.utils.registry import register_model_quantizer
from mqbench.prepare_by_platform import BackendType
from mqbench.custom_quantizer import ModelQuantizer
import torch.nn as nn
import mqbench.nn.qat as qnnqat
from collections import OrderedDict
from mqbench.utils.logger import logger

from typing import (
    List, Dict, Any, Callable
)

from mqbench.utils import get_flattened_qconfig_dict

@register_model_quantizer(BackendType.Sophgo_TPU)
class SophgoTpuQuantizer(ModelQuantizer):
    """There is only INT8 calculations in the model.
    We quantize the input tensors and output tensors of all layers,
    except those in _passed_func_type and _passed_module_type.
    For example add + relu pattern, there is no need to insert fake
    quantize node between them.
    """

    def __init__(self, extra_quantizer_dict, extra_fuse_dict):
        super().__init__(extra_quantizer_dict, extra_fuse_dict)
        self.additional_qat_module_mapping = {
            # Intrinsic modules:
            nni.ConvBn2d: qnniqat.ConvBn2d_sophgo,
            nni.ConvBnReLU2d: qnniqat.ConvBnReLU2d_sophgo,
            nn.Conv2d: qnnqat.Conv2d_sophgo,
            nni.ConvReLU2d: qnniqat.ConvReLU2d_sophgo,
            nni.LinearReLU: qnniqat.LinearReLU_sophgo,
            nn.Linear: qnniqat.Linear_sophgo,
            qnni.LinearBn1d: qnniqat.LinearBn1d_sophgo,
            qnni.ConvTransposeBnReLU2d:qnniqat.ConvTransposeBnReLU2d_sophgo,
            qnni.ConvTransposeReLU2d:qnniqat.ConvTransposeReLU2d_sophgo,
            qnni.ConvTransposeBn2d:qnniqat.ConvTransposeBn2d_sophgo,
        }
        self.exclude_module_name.append(nn.modules.dropout.Dropout)

    @property
    def module_type_to_quant_input(self) -> tuple:
        return (
            torch.nn.Hardswish,
            torch.nn.Sigmoid,
            torch.nn.SiLU,
            torch.nn.Tanh,
            torch.nn.SELU,
            torch.nn.LogSigmoid,
            torch.nn.GELU,
            torch.nn.GLU,
            torch.nn.Mish,
            torch.nn.Hardsigmoid,
            torch.nn.Softshrink,
            torch.nn.Softplus,
            torch.nn.ELU,
            torch.nn.CELU,            
        ) + super().module_type_to_quant_input + self._layers_need_scale_form_input_fake_quantizer

    @property
    def function_type_to_quant_input(self) -> tuple:
        return super().function_type_to_quant_input + [
            operator.sub,
            operator.abs,
            torch.cat,
            torch.sub,
            torch.clamp,
            torch.nn.functional.hardswish,
            torch.nn.functional.sigmoid,
            torch.nn.functional.silu,
            torch.nn.functional.tanh,
            torch.nn.functional.selu,
            torch.nn.functional.logsigmoid,
            torch.nn.functional.gelu,
            torch.nn.functional.glu,
            torch.nn.functional.mish,
            torch.nn.functional.hardsigmoid,
            torch.nn.functional.gumbel_softmax,
            torch.nn.functional.softshrink,
            torch.nn.functional.softplus,
            torch.nn.functional.elu,
            torch.nn.functional.celu,
        ]

    @property
    def _passed_func_type(self):
        return (
            torch.nn.functional.relu, 
            # torch.nn.functional.relu6, 
            torch.flatten
        )

    @property
    def _passed_module_type(self):
        return (
            torch.nn.ReLU,
            torch.nn.ReLU6
        )

    @property
    def _layers_need_scale_form_input_fake_quantizer(self):
        return (
            qnniqat.ConvBnReLU2d_sophgo, #todo:add transposeConv support
            qnniqat.ConvBn2d_sophgo, 
            qnniqat.ConvReLU2d_sophgo, 
            qnnqat.Conv2d_sophgo,
            qnniqat.LinearReLU_sophgo,
            qnniqat.Linear_sophgo,
            qnniqat.LinearBn1d_sophgo,
            qnniqat.ConvTransposeBnReLU2d_sophgo,
            qnniqat.ConvTransposeReLU2d_sophgo,
            qnniqat.ConvTransposeBn2d_sophgo,			
        )

    @property
    def _layers_need_check_is_dw(self):
        return (
            qnniqat.ConvBnReLU2d_sophgo,
            qnniqat.ConvBn2d_sophgo, 
            qnniqat.ConvReLU2d_sophgo, 
            qnnqat.Conv2d_sophgo,
        )
        
    def _insert_fake_quantizer(self, model, graph, modules, flattened_qconfig_dict, node, int84_layers):
        if len(int84_layers) > 0:
            layer = int84_layers[0] #多个int4或int8节点选其中第1个就能正确决定节点类型
            qconfig1 = flattened_qconfig_dict.get(layer.target, None) #首先根据层名去取，优先级最高
            if qconfig1 is None and layer.target in modules:
                print(f'layer.target:{layer.target}, type:',type(modules[layer.target]))
                qconfig1 = flattened_qconfig_dict.get(type(modules[layer.target]), None) #其次根据type去取
                if isinstance(modules[layer.target], self._layers_need_check_is_dw):
                    if modules[layer.target].groups > 1:
                        qconfig1 = None #深度卷积使用int8计算
            if qconfig1 is None:
                qconfig1 = flattened_qconfig_dict.get('', None) #最后找全局qconfig，优先级最低
            fake_quantizer = qconfig1.activation()
            quantizer_name = layer.name + self.quantizer_prefix
            if hasattr(model, quantizer_name):
                quantizer_name = layer.name +'_n2_br'+ self.quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for int84_layer in int84_layers:
                    int84_layer.args = self._fix_succ_recursivly(int84_layer.args, node, inserted_node)

    def _insert_fake_quantize_for_act_quant(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())

        self.quantizer_prefix = "_input_act_fake_quantizer"
        node_to_quantize_output = self._find_act_quants(model)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig)
        print('node_to_quantize_output:', node_to_quantize_output, 'flattened_qconfig_dict:', flattened_qconfig_dict)
        int4_and_int8_mix = False
        for m in flattened_qconfig_dict:
            if not isinstance(flattened_qconfig_dict[m].activation(), torch.nn.Identity) and flattened_qconfig_dict[m].activation().bitwidth == 4:
                int4_and_int8_mix = True
                break
        print('int4_and_int8_mix:', int4_and_int8_mix)
        if int4_and_int8_mix:
            def all_next_layers_is_trivial(node):
                if len(node.users) == 0:
                    return True
                for user in node.users:
                    if not all_next_layers_is_trivial(user):
                        return False
                if node.op != 'call_method' or node.target not in  ['view', 'permute', 'contiguous']:
                    return False
                return True
                        
            def find_next_int4_and_int8_layers(node, int8_layers, int4_layers):
                for user in node.users: #若后继有1个层或多个不同类型的层，则插入多个input量化节点  todo:多个节点，部分相同，部分不同
                    if user.target in modules and type(modules[user.target]) in self.exclude_module_name:
                        print(f'user:{user.name} is excluded')
                        user.replace_all_uses_with(node)
                        graph.erase_node(user)
                        del modules[user.target]
                        find_next_int4_and_int8_layers(user, int8_layers, int4_layers)
                        continue #dropout等层前不要插入伪量化节点
                    if user.op == "call_module" and isinstance(modules[user.target], self._layers_need_scale_form_input_fake_quantizer):
                        if isinstance(modules[user.target], self._layers_need_check_is_dw):
                            if modules[user.target].groups > 1:
                                int8_layers.append(user)
                                continue
                        int4_layers.append(user)
                    else:
                        if not all_next_layers_is_trivial(user):
                            int8_layers.append(user)

            for node in node_to_quantize_output:
                int8_layers, int4_layers = [],[] #找到node后的多个int4后继节点和多个int8后继节点，然后这多个int8或int4后继节点共享1个输入量化节点
                find_next_int4_and_int8_layers(node, int8_layers, int4_layers)
                print(f'node:{node}, int4_layers:', int4_layers, 'int8_layers:', int8_layers)
                self._insert_fake_quantizer(model, graph, modules, flattened_qconfig_dict, node, int4_layers)
                self._insert_fake_quantizer(model, graph, modules, flattened_qconfig_dict, node, int8_layers)
            model.recompile()
            model.graph.lint()
            graph = model.graph
            nodes = list(model.graph.nodes)
            modules = dict(model.named_modules())

        for node in node_to_quantize_output:
            if node.op == 'placeholder' and int4_and_int8_mix:
                continue
            qconfig2 = flattened_qconfig_dict.get(node.target, None) #首先根据层名去取，优先级最高
            if qconfig2 is None and node.target in modules:
                qconfig2 = flattened_qconfig_dict.get(type(modules[node.target]), None) #其次根据type去取
                if isinstance(modules[node.target], self._layers_need_check_is_dw):
                    if modules[node.target].groups > 1:
                        qconfig2 = None #深度卷积使用int8计算
            if qconfig2 is None:
                qconfig2 = flattened_qconfig_dict.get('', None) #最后找全局qconfig，优先级最低
            if int4_and_int8_mix:
                if node.target in modules and type(modules[node.target]) == torch.nn.ReLU6 and node.args[0].target in modules:
                    qconfig2 = flattened_qconfig_dict.get(type(modules[node.args[0].target]), None)
                    if isinstance(modules[node.args[0].target], self._layers_need_check_is_dw):
                        if modules[node.args[0].target].groups > 1:
                            qconfig2 = flattened_qconfig_dict.get('', None) ##深度卷积使用int8计算
            node_fake_quantizer = qconfig2.activation()
            # node_fake_quantizer.enable_only_observer()
            quantizer_name2 = node.name + "_post_act_fake_quantizer"
            setattr(model, quantizer_name2, node_fake_quantizer)
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name2, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        if int4_and_int8_mix:
            model.recompile()
            model.graph.lint()
            graph = model.graph
            modules = dict(model.named_modules())
            nodes = list(model.graph.nodes)
            for node in nodes:
                if "_post_act_fake_quantizer" in node.name:
                    #post量化节点的下一个是input量化节点，且input量化节点的后面只有1个节点，且该节点后再无后继节点，此时删除这个input量化节点，用于删除网络输出的最后1个多余的量化节点
                    for user in list(node.users.keys()):
                        if "_input_act_fake_quantizer" in user.name: 
                            users = list(user.users.keys())
                            if len(users) == 1 and len(users[0].users) == 0:
                                user.replace_all_uses_with(node)
                                graph.erase_node(user)
                                del modules[user.target]
                    if modules[node.target].bitwidth == 8: #2个相邻的量化节点都为8bit量化，则删除冗余的后1个节点
                        for user in list(node.users.keys()):
                            if "_input_act_fake_quantizer" in user.name:
                                if modules[user.target].bitwidth == 8:
                                    user.replace_all_uses_with(node)
                                    graph.erase_node(user)
                                    del modules[user.target]
                    elif modules[node.target].bitwidth == 4: #2个相邻的量化节点都为4bit量化，则删除冗余的后1个节点
                        for user in list(node.users.keys()):
                            if "_input_act_fake_quantizer" in user.name:
                                if modules[user.target].bitwidth == 4:
                                    user.replace_all_uses_with(node)
                                    graph.erase_node(user)
                                    del modules[user.target]
        model.recompile()
        model.graph.lint()        
        return model

    def prepare(self, model: GraphModule, qconfig):
        model = super().prepare(model, qconfig)
        self._show_all_fake_quant_name(model)
        model = self._set_fake_quantizer_to_next_weight_layer(model)
        model = self._set_module_only_enable_observer(model)
        return model

    def _find_act_quants(self, model: GraphModule) -> list:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = super()._find_act_quants(model)
        self.only_enable_ob = []
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
                    if ((next_node.op == 'call_function' and next_node.target in self._passed_func_type) or
                            (next_node.op == 'call_module' and isinstance(modules[next_node.target], self._passed_module_type))):
                        if next_node not in node_need_to_quantize_output:
                            node_need_to_quantize_output.append(next_node)
                            self.only_enable_ob.append(next_node.name)
                    else:
                        if node not in node_need_to_quantize_output:
                            node_need_to_quantize_output.append(node)
                            self.only_enable_ob.append(node.name)
        for node in nodes:
            if node.target in modules and type(modules[node.target]) in self.exclude_module_name:
            # if node.target in modules and (type(modules[node.target]) in self.exclude_module_name or node.target in self.exclude_module_name) and (node in node_need_to_quantize_output):
                print(f'{type(modules[node.target])} is excluded')
                node_need_to_quantize_output.remove(node)
            if node.op == "placeholder" and (node.name not in self.exclude_node_name):
                if 'tensor_meta' in node.meta:
                    if len(node.meta['tensor_meta'].shape) > 1:
                        print(f'add placeholder {node.target} to node_need_to_quantize_output by tensor_meta')
                        node_need_to_quantize_output.append(node)
                else:
                    print(f'no tensor_meta, add placeholder {node.target} to node_need_to_quantize_output')
                    node_need_to_quantize_output.append(node)

        return node_need_to_quantize_output
        
    def _set_fake_quantizer_to_next_weight_layer(self, model: GraphModule):
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        for node in nodes:
            if node.target in modules and (self.quantizer_prefix in node.target or "_post_act_fake_quantizer" in node.target):
                fake_quantizer = getattr(model, node.target)
                for user in node.users:
                    if (user.op == "call_module" and isinstance(modules[user.target], self._layers_need_scale_form_input_fake_quantizer)):
                        setattr(modules[user.target], "input_fake_quantizer", fake_quantizer)
                        print('wlog:', user.target,'\'type is:', type(modules[user.target]), "add input_fake_quantizer")
                    if user.target in modules and type(modules[user.target]) in self.exclude_module_name:
                        for user2 in user.users:
                            if (user2.op == "call_module" and isinstance(modules[user2.target], self._layers_need_scale_form_input_fake_quantizer)):
                                setattr(modules[user2.target], "input_fake_quantizer", fake_quantizer)
                                print('wlog:', user2.target,'\'type is:', type(modules[user2.target]), "add input_fake_quantizer")

        return model

    def _show_all_fake_quant_name(self, model: GraphModule):
        print(">>>>> All fake quant nodes have been inserted. Their names : ")
        for name, submodule in model.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                print(">>>>> ", name)
        print(">>>>> end")
        return

    def _set_module_only_enable_observer(self, model: GraphModule):
        if len(self.module_only_enable_observer) == 0:
            return model
        for name, submodule in model.named_modules(remove_duplicate=False):
            if isinstance(submodule, torch.quantization.FakeQuantizeBase) and (name in self.module_only_enable_observer):
                logger.info('Only enable observer : {}'.format(name))
                submodule.enable_observer()
                submodule.disable_fake_quant()
                submodule.only_enable_observer = True
                # setattr(submodule, 'only_enable_ob', True)
                # import pdb;pdb.set_trace()
        return model