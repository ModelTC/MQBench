from typing import Dict, Any
import torch
from torch.fx.graph import Graph
from torch.fx import GraphModule, map_arg
from torch.quantization.fx.fuse import Fuser  # noqa: F401
from torch.quantization.utils import get_combined_dict
from torch.quantization.fx.pattern_utils import get_default_fusion_patterns


class CustomFuser(Fuser):
    def fuse(self, model: GraphModule,
             fuse_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        input_root = model
        input_graph = model.graph
        self.modules = dict(input_root.named_modules())

        additional_fusion_patterns = \
            fuse_custom_config_dict.get("additional_fusion_pattern", {})
        fusion_patterns = get_combined_dict(
            get_default_fusion_patterns(), additional_fusion_patterns)
        # find fusion
        fusion_pairs = self._find_matches(
            input_root, input_graph, fusion_patterns)
        self.fused_graph = Graph()
        env: Dict[Any, Any] = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in input_graph.nodes:
            root_node, obj = fusion_pairs.get(node.name, (None, None))
            if root_node is node:
                assert obj is not None
                env[node.name] = obj.fuse(self, load_arg, fuse_custom_config_dict=fuse_custom_config_dict)
            elif root_node is None:
                env[node.name] = self.fused_graph.node_copy(node, load_arg)
            # node matched in patterns and is not root is removed here

        model = GraphModule(input_root, self.fused_graph)
        return model


def _check_is_graph_module(model: torch.nn.Module) -> None:
    if not isinstance(model, GraphModule):
        raise ValueError(
            'input model must be a GraphModule, ' +
            'Got type:' + str(type(model)) + ' Please make ' +
            'sure to follow the tutorials.')

def fuse_fx(
        graph_module: GraphModule,
        fuse_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
    r""" Internal helper function to fuse modules in preparation for quantization

    Args:
        graph_module: GraphModule object from symbolic tracing (torch.fx.symbolic_trace)
    """
    _check_is_graph_module(graph_module)
    fuser = CustomFuser()
    return fuser.fuse(graph_module, fuse_custom_config_dict)