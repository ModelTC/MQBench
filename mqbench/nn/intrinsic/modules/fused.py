
from torch.nn.intrinsic import _FusedModule
from torch.nn import Linear, BatchNorm1d


class LinearBn1d(_FusedModule):
    r"""This is a sequential container which calls the Linear and Batch Norm 1d modules.
    During quantization this will be replaced with the corresponding fused module."""
    def __init__(self, linear, bn):
        assert type(linear) == Linear and type(bn) == BatchNorm1d, \
            'Incorrect types for input modules{}{}'.format(
                type(linear), type(bn))
        super().__init__(linear, bn)