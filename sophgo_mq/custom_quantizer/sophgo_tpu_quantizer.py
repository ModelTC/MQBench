import operator
import torch
from torch.fx import GraphModule
import torch.nn.intrinsic as nni
from sophgo_mq.utils import getitem2node
import sophgo_mq.nn.intrinsic.qat as qnniqat
import sophgo_mq.nn.intrinsic as qnni
from sophgo_mq.utils.registry import register_model_quantizer
from sophgo_mq.custom_quantizer import ModelQuantizer
import torch.nn as nn
import sophgo_mq.nn.qat as qnnqat
from collections import OrderedDict
from sophgo_mq.utils.logger import logger

from sophgo_mq.fake_quantize import (
    LearnableFakeQuantize,
    Fp16FakeQuantize,
    BF16FakeQuantize
)
from typing import (
    List, Dict, Any, Callable
)
from sophgo_mq.utils import get_flattened_qconfig_dict


@register_model_quantizer("BM1688")
@register_model_quantizer("BM1690")
@register_model_quantizer("BM1684X")
@register_model_quantizer("CV183X")
class SophgoTpuQuantizer(ModelQuantizer):
    """
    We quantize the input tensors and output tensors of all layers,
    except those in _passed_func_type and _passed_module_type.
    For example add + relu pattern, there is no need to insert fake
    quantize node between them.
    """
    def __init__(self, extra_quantizer_dict, extra_fuse_dict,quant_dict,chip):
        super().__init__(extra_quantizer_dict, extra_fuse_dict,quant_dict)
        if self.quantmode=="weight_activation":
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
        self.chip = chip

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
            torch.nn.LayerNorm,
            torch.nn.qat.modules.linear.Linear,
        ) + super().module_type_to_quant_input + self._layers_need_scale_form_input_fake_quantizer

    @property
    def function_type_to_quant_input(self) -> tuple:
        return super().function_type_to_quant_input + [
            operator.sub,
            operator.abs,
            torch.cat,
            torch.sub,
            torch.clamp,
            torch.matmul,
            operator.add,
            operator.truediv,
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
            torch.nn.functional.layer_norm,
        ]

    @property
    def function_type_to_quant_input_transformer(self) -> list:
        return [
            # Matmul in MSA
            torch.matmul
        ] + self.additional_function_type

    @property
    def module_type_to_quant_input_transformer(self) -> tuple:
        return (
            # Linear
            qnnqat.Conv2d_sophgo,
            qnniqat.LinearReLU_sophgo,
            qnniqat.Linear_sophgo,
            torch.nn.qat.modules.linear.Linear,
        ) + self.additional_module_type

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


    def _insert_fake_quantizer(self, model, graph, modules, flattened_qconfig_dict, node, next_layers, fp16 = False):
        if len(next_layers) > 0:
            layer = next_layers[0] #选其中第1个就能正确决定节点类型
            qconfig1 = flattened_qconfig_dict.get(layer.target, None) #首先根据层名去取，优先级最高
            if qconfig1 is None and layer.target in modules:
                print(f'layer.target:{layer.target}, type:',type(modules[layer.target]))
                qconfig1 = flattened_qconfig_dict.get(type(modules[layer.target]), None) #其次根据type去取
                if isinstance(modules[layer.target], self._layers_need_check_is_dw):
                    if modules[layer.target].groups > 1:
                        qconfig1 = None
            if qconfig1 is None:
                qconfig1 = flattened_qconfig_dict.get('', None) #最后找全局qconfig，优先级最低
            fake_quantizer = qconfig1.activation()
            if fp16:
                print(f'insert bf16 node')
                fake_quantizer = BF16FakeQuantize(None)
            quantizer_name = layer.name + self.quantizer_prefix
            if hasattr(model, quantizer_name):
                quantizer_name = layer.name +'_n2_br'+ self.quantizer_prefix
            setattr(model, quantizer_name, fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name, (node,), {})
                for next_layer in next_layers:
                    next_layer.args = self._fix_succ_recursivly(next_layer.args, node, inserted_node)

    def _insert_fake_quantize_for_act_quant(
            self,
            model: GraphModule,
            qconfig: Any):
        graph = model.graph
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())

        self.quantizer_prefix = "_input_act_fake_quantizer"
        user_specified_layers_need_to_quant = self._layers_need_scale_form_input_fake_quantizer if self.chip == 'CV183X' else None
        node_to_quantize_output = self._find_act_quants(model, user_specified_layers_need_to_quant)
        self.node_to_quantize_output_transformer=self._find_act_quants_transformer(model)
        node_to_quantize_output.extend(node for node in self.node_to_quantize_output_transformer if node not in node_to_quantize_output)
        node_to_quantize_output = OrderedDict.fromkeys(node_to_quantize_output).keys()
        flattened_qconfig_dict = get_flattened_qconfig_dict(qconfig)
        logger.info('node_to_quantize_output:{}'.format(node_to_quantize_output))
        logger.info('flattened_qconfig_dict:{}'.format(flattened_qconfig_dict))

        if self.chip == "CV183X":
            def find_next_f16_and_int8_layers(node, int8_layers, f16_layers):
                for user in node.users: #若后继有1个层或多个不同类型的层，则插入多个input量化节点  todo:多个节点，部分相同，部分不同
                    if user.target in modules and type(modules[user.target]) in self.exclude_module_name:
                        print(f'user:{user.name} is excluded')
                        user.replace_all_uses_with(node)
                        graph.erase_node(user)
                        del modules[user.target]
                        find_next_f16_and_int8_layers(user, int8_layers, f16_layers)
                        continue #dropout等层前不要插入伪量化节点
                    if user.op == "call_module" and isinstance(modules[user.target], self._layers_need_scale_form_input_fake_quantizer):
                        int8_layers.append(user)
                    else:
                        f16_layers.append(user)

            for node in node_to_quantize_output:
                int8_layers, f16_layers = [],[] #找到node后的多个int4后继节点和多个int8后继节点，然后这多个int8或int4后继节点共享1个输入量化节点
                find_next_f16_and_int8_layers(node, int8_layers, f16_layers)
                print(f'node:{node}, f16_layers:', f16_layers, 'int8_layers:', int8_layers)
                self._insert_fake_quantizer(model, graph, modules, flattened_qconfig_dict, node, f16_layers, True)
                self._insert_fake_quantizer(model, graph, modules, flattened_qconfig_dict, node, int8_layers)
            model.recompile()
            model.graph.lint()
            graph = model.graph
            nodes = list(model.graph.nodes)
            modules = dict(model.named_modules())

        quantizer_prefix = "_post_act_fake_quantizer"
        for node in node_to_quantize_output:
            qconfig2 = flattened_qconfig_dict.get(node.name, None) #首先根据node.name去取，优先级最高
            if qconfig2 is None and node.target in modules:
                qconfig2 = flattened_qconfig_dict.get(type(modules[node.target]), None) #其次根据type去取
                if isinstance(modules[node.target], self._layers_need_check_is_dw):
                    if modules[node.target].groups > 1:
                        qconfig2 = None #深度卷积使用int8计算
            if qconfig2 is None:
                qconfig2 = flattened_qconfig_dict.get('', None) #最后找全局qconfig，优先级最低
            node_fake_quantizer = qconfig2.activation()
            if self.chip == "CV183X" and ((node.op == "call_module" and not isinstance(modules[node.target], self._layers_need_scale_form_input_fake_quantizer))
                    or node.op == "placeholder"):
                print(f'insert BF16FakeQuantize for output of {node.name}')
                node_fake_quantizer = BF16FakeQuantize(None)
            quantizer_name2 = node.name + quantizer_prefix
            setattr(model, quantizer_name2, node_fake_quantizer)
            logger.info("Insert act quant {}".format(quantizer_name2))
            with graph.inserting_after(node):
                inserted_node = graph.create_node("call_module", quantizer_name2, (node,), {})
                for _node in nodes:
                    _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)

        if self.chip == "CV183X":
            model.recompile()
            model.graph.lint()
            graph = model.graph
            modules = dict(model.named_modules())
            nodes = list(model.graph.nodes)
            for node in nodes:
                if "_post_act_fake_quantizer" in node.name:
                    if isinstance(modules[node.target], BF16FakeQuantize): #2个相邻的量化节点都为BF16FakeQuantize，则删除冗余的后1个节点
                        for user in list(node.users.keys()):
                            if "_input_act_fake_quantizer" in user.name:
                                if isinstance(modules[user.target], BF16FakeQuantize):
                                    user.replace_all_uses_with(node)
                                    graph.erase_node(user)
                                    del modules[user.target]
                                    print(f'del {user.name}, {node.name} left')
                    elif isinstance(modules[node.target], LearnableFakeQuantize):
                        for user in list(node.users.keys()):
                            if "_input_act_fake_quantizer" in user.name:
                                if isinstance(modules[user.target], LearnableFakeQuantize):
                                    user.replace_all_uses_with(node)
                                    graph.erase_node(user)
                                    del modules[user.target]
                                    print(f'del {user.name}, {node.name} left')
            model.recompile()
            model.graph.lint()
            graph = model.graph
            modules = dict(model.named_modules())
            nodes = list(model.graph.nodes)
            bf16_layer_type = (torch.nn.modules.pooling.AdaptiveAvgPool2d)
            for node in nodes:
                if (node.op == "call_module" and isinstance(modules[node.target], self._layers_need_scale_form_input_fake_quantizer)):
                    continue
                if (node.op == "call_module" and isinstance(modules[node.target], bf16_layer_type)):
                    node_fake_quantizer = BF16FakeQuantize(None)
                    quantizer_name2 = node.name + "_post_act_fake_bf16_quantizer"
                    setattr(model, quantizer_name2, node_fake_quantizer)
                    logger.info("Insert out act bf16 quant {}".format(quantizer_name2))
                    self.insert_node(model, node, quantizer_name2)
                    input_node_list = self._flatten_args(node.all_input_nodes)     
                    for n in input_node_list:
                        valid, n = self.is_valid_fx_input_arg(model, node, n)
                        if valid:
                            node_fake_quantizer = BF16FakeQuantize(None)
                            quantizer_name2 = node.name + "_input_act_fake_bf16_quantizer"
                            setattr(model, quantizer_name2, node_fake_quantizer)
                            logger.info("Insert input act bf16 quant {}".format(quantizer_name2))
                            self.insert_node(model, n, quantizer_name2)
        model.recompile()
        model.graph.lint()        
        return model

    def prepare(self, model: GraphModule, qconfig):
        model = super().prepare(model, qconfig)
        self._show_all_fake_quant_name(model)
        if self.quantmode=="weight_activation":
            model = self._set_fake_quantizer_to_next_weight_layer(model)
            model = self._set_module_only_enable_observer(model)
        return model
    
    def insert_node(self, model, node, new_node_name):
        graph = model.graph
        nodes = list(model.graph.nodes)
        with graph.inserting_after(node):
            inserted_node = graph.create_node("call_module", new_node_name, (node,), {})
            for _node in nodes:
                _node.args = self._fix_succ_recursivly(_node.args, node, inserted_node)    

    def insert_bf16_node(self, model: GraphModule):
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        for node in nodes:
            if '_post_act_fake_quantizer' in node.name:
                continue
            if ((node.op == "call_module" and not isinstance(modules[node.target], self._layers_need_scale_form_input_fake_quantizer )) or \
                ((node.op == 'call_function' or node.op == 'call_method') and node.target in self.function_type_to_quant_input)):
                node_fake_quantizer = BF16FakeQuantize(None)
                quantizer_name2 = node.name + "_post_act_fake_bf16_quantizer"
                setattr(model, quantizer_name2, node_fake_quantizer)
                logger.info("Insert out act bf16 quant {}".format(quantizer_name2))
                self.insert_node(model, node, quantizer_name2)
                input_node_list = self._flatten_args(node.all_input_nodes)     
                for n in input_node_list:
                    valid, n = self.is_valid_fx_input_arg(model, node, n)
                    if valid:
                        node_fake_quantizer = BF16FakeQuantize(None)
                        quantizer_name2 = node.name + "_input_act_fake_bf16_quantizer"
                        setattr(model, quantizer_name2, node_fake_quantizer)
                        logger.info("Insert input act bf16 quant {}".format(quantizer_name2))
                        self.insert_node(model, n, quantizer_name2)
        model.recompile()
        model.graph.lint()    
        return model    

    def _find_act_quants(self, model: GraphModule, user_specified_layers_need_to_quant = None) -> list:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = super()._find_act_quants(model, user_specified_layers_need_to_quant)
        module_type_to_quant_input = self.module_type_to_quant_input if user_specified_layers_need_to_quant is None else user_specified_layers_need_to_quant
        function_type_to_quant_input = self.function_type_to_quant_input if user_specified_layers_need_to_quant is None else []
        self.only_enable_ob = []
        if self.strategy == 'Transformer':
            for node in nodes:
                if node.name in ['qa_outputs','squeeze','squeeze_1']:
                    node_need_to_quantize_output.append(node)
        if self.strategy == 'CNN':
            for node in nodes:
                if (node.op == "call_module" and node.target in self.exclude_module_name) or \
                    ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.exclude_function_type) or \
                        node.name in self.exclude_node_name:
                    continue
                if (node.op == "call_module" and isinstance(modules[node.target], module_type_to_quant_input)) or \
                    ((node.op == 'call_function' or node.op == 'call_method') and
                        node.target in function_type_to_quant_input):
                    for next_node in node.users:
                        if ((next_node.op == 'call_function' and next_node.target in self._passed_func_type) or
                                (next_node.op == 'call_module' and isinstance(modules[next_node.target], self._passed_module_type))):
                            if next_node not in node_need_to_quantize_output:
                                node_need_to_quantize_output.append(next_node)
                                print(">>>>> append node1 ", next_node)
                                self.only_enable_ob.append(next_node.name)
                        else:
                            if node not in node_need_to_quantize_output:
                                node_need_to_quantize_output.append(node)
                                print(">>>>> append node2 ", node)
                                self.only_enable_ob.append(node.name)
        for node in nodes:
            # if node.target in modules and (type(modules[node.target]) in self.exclude_module_name or node.target in self.exclude_module_name) and (node in node_need_to_quantize_output):
            #     print(f'{type(modules[node.target])} is excluded')
            #     node_need_to_quantize_output.remove(node)
            if node.op == "placeholder" and (node.name not in self.exclude_node_name):
                if 'tensor_meta' in node.meta:
                    if len(node.meta['tensor_meta'].shape) > 1:
                        print(f'add placeholder {node.target} to node_need_to_quantize_output by tensor_meta')
                        node_need_to_quantize_output.append(node)
                        print(">>>>> append node ", node)
                else:
                    print(f'no tensor_meta, add placeholder {node.target} to node_need_to_quantize_output')
                    node_need_to_quantize_output.append(node)
                    print(">>>>> append node ", node)
        
        return node_need_to_quantize_output

    def _find_act_quants_transformer(self, model: GraphModule) -> List:
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        node_need_to_quantize_output = []
        g2node = getitem2node(model)
        for node in nodes:
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                logger.info("Exclude skip: {}".format(node.name))
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input_transformer)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input_transformer) or node.name in self.additional_node_name:
                input_node_list = self._flatten_args(node.all_input_nodes)
                # Means this is not Tensor + Tensor.
                if not all([isinstance(_node, torch.fx.node.Node) for _node in input_node_list]):
                    continue
                for _node in input_node_list:
                    if self._is_implicit_merge(modules, (node, _node)):
                        logger.info("Implicit merge: {} + {}".format(_node.name, node.name))
                        continue
                    if _node.op == "placeholder" and 'tensor_meta' in node.meta:
                        if len(_node.meta['tensor_meta'].shape) == 1:
                            continue
                    if _node in node_need_to_quantize_output:
                        continue
                    if _node in g2node:
                        _node = g2node[_node]
                    node_need_to_quantize_output.append(_node)
        return node_need_to_quantize_output

    def _set_fake_quantizer_to_next_weight_layer(self, model: GraphModule):
        nodes = list(model.graph.nodes)
        modules = dict(model.named_modules())
        self.quantizer_prefix = "_input_act_fake_quantizer"
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
        logger.info(">>>>> All fake quant nodes have been inserted. Their names : ")
        for name, submodule in model.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                logger.info("{}".format(name))
        logger.info(">>>>> end")
        return

    def _set_module_only_enable_observer(self, model: GraphModule):
        if self.strategy=="Transformer":
            module_only_transformer=[]
            for node in self.node_to_quantize_output_transformer:
                module_only_transformer.append(node.name)
            for name, submodule in model.named_modules(remove_duplicate=False):
                if isinstance(submodule, torch.quantization.FakeQuantizeBase) and "_post_act_fake_quantizer"in name:
                    layer_name=name.split("_post_act_fake_quantizer")[0]
                    if layer_name not in module_only_transformer:
                        logger.info('Only enable observer : {}'.format(name))
                        submodule.only_enable_observer = True
        for name, submodule in model.named_modules(remove_duplicate=False):
            if isinstance(submodule, torch.quantization.FakeQuantizeBase) and (name in self.module_only_enable_observer):
                logger.info('Only enable observer : {}'.format(name))
                submodule.enable_observer()
                submodule.disable_fake_quant()
                submodule.only_enable_observer = True
        return model