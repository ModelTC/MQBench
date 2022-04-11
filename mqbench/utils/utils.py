import copy

import torch
from torch.fx import GraphModule

USE_LINK = False
USE_DDP = False

try:
    import spring.linklink as link
    assert link.is_initialized()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True


def sync_tensor(tensor):
    global USE_LINK
    global USE_DDP
    if USE_LINK:
        if tensor.is_cuda is True:
            tensor.data = tensor.data / link.get_world_size()
            link.allreduce(tensor.data)
    elif USE_DDP:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


def pot_quantization(tensor: torch.Tensor, mode='round'):
    log2t = torch.log2(tensor)
    if mode == 'round':
        log2t = (torch.round(log2t) - log2t).detach() + log2t
    else:
        assert mode == 'floor' 
        log2t = (torch.floor(log2t) - log2t).detach() + log2t
    return 2 ** log2t



def is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


class no_jit_trace:
    def __enter__(self):
        # pylint: disable=protected-access
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)

    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def is_tracing_state():
    return torch._C._get_tracing_state()


def deepcopy_graphmodule(gm: GraphModule):
    """Rewrite the deepcopy of GraphModule. (Copy its 'graph'.)

    Args:
        gm (GraphModule): 

    Returns:
        GraphModule: A deepcopied gm.
    """
    copied_gm = copy.deepcopy(gm)
    copied_gm.graph = copy.deepcopy(gm.graph)
    return copied_gm


def getitem2node(model: GraphModule) -> dict:
    def _update_getitem_path(getitem_args_dict):
        for node in getitem_args_dict:
            args_list = getitem_args_dict[node]
            while args_list[0] in getitem_args_dict:
                args_list = getitem_args_dict[args_list[0]] + args_list[1:]
            getitem_args_dict[node] = args_list
        return getitem_args_dict

    def _getitem_from_args(args, original_args_dict):
        ret = original_args_dict
        for a in args:
            try:
                ret = ret[a]
            except (IndexError, KeyError):
                return {}
        return ret 
    import operator
    nodes = list(model.graph.nodes)
    # the getitem's call graph
    getitem_args_dict = {}
    # the dict used in the model 
    original_key_dict = {}
    getitem2node = {}
    for node in nodes:
        # update the getitems
        if node.target == operator.getitem:
            getitem_args_dict[node] = list(node.args)
            getitem_args_dict = _update_getitem_path(getitem_args_dict)
        elif node.target == 'update':
            if node.args[0] not in original_key_dict:
                original_key_dict[node.args[0]] = {}
            original_key_dict[node.args[0]].update(node.args[1])
    for node in getitem_args_dict:
        val = _getitem_from_args(getitem_args_dict[node], original_key_dict)
        if isinstance(val, torch.fx.node.Node):
            getitem2node[node] = val
    return getitem2node


def topology_order(model):
    node2idx = {}
    for idx, node in enumerate(model.graph.nodes):
        node2idx[node] = idx 
    return node2idx