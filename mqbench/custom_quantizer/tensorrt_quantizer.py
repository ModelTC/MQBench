import operator

import torch
from torch.fx import GraphModule

from mqbench.custom_quantizer import ModelQuantizer


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
            if ((node.op == "call_module" and node.target in self.exclude_module_name) or
                ((node.op == 'call_function' or node.op == 'call_method') and
                 node.target in self.exclude_function_type) or
                    node.name in self.exclude_node_name) and node.name not in self.additional_node_name:
                continue
            if (node.op == "call_module" and isinstance(modules[node.target], self.module_type_to_quant_input)) or \
                ((node.op == 'call_function' or node.op == 'call_method') and
                    node.target in self.function_type_to_quant_input) or node.name in self.additional_node_name:
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