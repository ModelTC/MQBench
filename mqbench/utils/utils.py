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
    import torch.distributed
    if torch.distributed.is_initialized():
        USE_DDP = True


def sync_tensor(tensor):
    global USE_LINK
    global USE_DDP
    if USE_LINK:
        if tensor.is_cuda is True:
            world_size = link.get_world_size()
            link.allreduce(tensor / world_size)
    elif USE_DDP:
        world_size = dist.get_world_size()
        dist.allreduce(tensor / world_size)


def pot_quantization(tensor: torch.Tensor):
    log2t = torch.log2(tensor)
    log2t = (torch.round(log2t) - log2t).detach() + log2t
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
