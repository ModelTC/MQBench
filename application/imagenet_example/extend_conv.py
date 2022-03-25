from typing import Any, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import torch.nn.intrinsic.qat as torch_qat
from torch.nn.intrinsic import _FusedModule

from torch.fx.graph import Node
from torch.quantization.fx.fusion_patterns import FuseHandler, QuantizerCls
from torch.quantization.fx.utils import _parent_name
from torch.quantization.fuser_method_mappings import get_fuser_method

from mqbench.utils.registry import register_convert_function

from efficientnet import Conv2dStaticSamePadding




class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class ConvStaticSamePaddingBn2d(_FusedModule):
    def __init__(self, conv, bn):
        assert type(conv) == Conv2dStaticSamePadding and type(bn) == nn.BatchNorm2d, \
            'Incorrect types for input modules{}{}'.format(
                type(conv), type(bn))
        if type(conv.static_padding) == nn.ZeroPad2d:
            pad = nn.ZeroPad2d(conv.static_padding.padding)
        else:
            pad = Identity()
        super().__init__(pad, conv, bn)


class ConvStaticSamePaddingBn2dFusion(FuseHandler):
    def __init__(self, quantizer, node):
        super(ConvStaticSamePaddingBn2dFusion, self).__init__(quantizer, node)
        self.bn_node = None
        if type(quantizer.modules[node.target]) in [torch.nn.BatchNorm2d]:
            self.bn_node = node
            self.bn = quantizer.modules[self.bn_node.target]
            assert isinstance(node.args[0], Node)
            node = node.args[0]
        assert node.op == 'call_module'
        self.conv_node = node
        self.conv = quantizer.modules[self.conv_node.target]
    
    def fuse(self, quantizer: QuantizerCls, load_arg: Callable,
             fuse_custom_config_dict: Dict[str, Any] = None) -> Node:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get("additional_fuser_method_mapping", {})
        op_list = []
        
        assert self.bn_node is not None
        op_list.append(self.bn)
        op_list.append(self.conv)

        # the modules are added in order of relu - bn - conv
        # so we need to correct it
        op_list.reverse()
        op_type_list = tuple(type(m) for m in op_list)
        conv_parent_name, conv_name = _parent_name(self.conv_node.target)
        fuser_method = get_fuser_method(op_type_list, additional_fuser_method_mapping)
        if fuser_method is None:
            raise NotImplementedError("Cannot fuse modules: {}".format(op_type_list))
        fused = fuser_method(*op_list)
        setattr(quantizer.modules[conv_parent_name], conv_name, fused)

        # TODO: do we need to make sure bn is only used once?
        if self.bn_node is not None:
            parent_name, name = _parent_name(self.bn_node.target)
            setattr(quantizer.modules[parent_name], name, torch.nn.Identity())
        # relu may be used multiple times, so we don't set relu to identity
        return quantizer.fused_graph.node_copy(self.conv_node, load_arg)


def fuse_conv_static_same_padding_bn(conv, bn):
    assert(conv.training == bn.training),\
        'DeConv and BN must be in the same mode (train or eval)'

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return ConvStaticSamePaddingBn2d(conv, bn)
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)



class QConvStaticSamePaddingConvBn2d(torch_qat.ConvBn2d, nn.Conv2d):
    r"""
    A ConvBn2d module is a module fused from Conv2d and BatchNorm2d,
    attached with FakeQuantize modules for weight,
    used in quantization aware training.

    We combined the interface of :class:`torch.nn.Conv2d` and
    :class:`torch.nn.BatchNorm2d`.

    Similar to :class:`torch.nn.Conv2d`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """
    _FLOAT_MODULE = ConvStaticSamePaddingBn2d
    
    def forward(self, input):
        return self._forward(self.static_padding(input))

    @classmethod
    def from_float(cls, mod):
        r"""Create a qat module from a float module or qparams_dict

            Args: `mod` a float module, either produced by torch.quantization utilities
            or directly from user
        """
        assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
            cls._FLOAT_MODULE.__name__
        assert hasattr(mod, 'qconfig'), 'Input float module must have qconfig defined'
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        static_padding, conv, bn = mod[0], mod[1], mod[2]
        qat_convbn = cls(conv.in_channels, conv.out_channels, conv.kernel_size,
                         conv.stride, conv.padding, conv.dilation,
                         conv.groups, conv.bias is not None,
                         conv.padding_mode,
                         bn.eps, bn.momentum,
                         False,
                         qconfig)
        qat_convbn.weight = conv.weight
        qat_convbn.bias = conv.bias
        qat_convbn.bn.weight = bn.weight
        qat_convbn.bn.bias = bn.bias
        qat_convbn.bn.running_mean = bn.running_mean
        qat_convbn.bn.running_var = bn.running_var
        # mypy error: Cannot determine type of 'num_batches_tracked'
        qat_convbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[has-type]
        qat_convbn.static_padding = static_padding

        return qat_convbn


class QConvStaticSamePadding2d(nn.Conv2d):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1, bias=True, dilation=1,
                 padding_mode='zeros', qconfig=None):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding, groups=groups, 
                         bias=bias, dilation=dilation, padding_mode=padding_mode)
        assert qconfig, 'qconfig must be provided for QAT module'
        self.qconfig = qconfig 
        self.weight_fake_quant = qconfig.weight()
        # if self.weight_fake_quant.ch_axis != -1:
        #     self.weight_fake_quant.ch_axis = 0
        #     self.weight_fake_quant.activation_post_process.ch_axis = 1

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight_fake_quant(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x

    @classmethod
    def from_float(cls, mod):
        # assert type(mod) == cls._FLOAT_MODULE, 'qat.' + cls.__name__ + '.from_float only works for ' + \
        #     cls._FLOAT_MODULE.__name__ 
        assert mod.qconfig, 'Input float module must have a valid qconfig'
        qconfig = mod.qconfig
        qat_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, 
                         stride=mod.stride, padding=mod.padding,
                         groups=mod.groups, bias=mod.bias is not None, dilation=mod.dilation,
                         padding_mode=mod.padding_mode, qconfig=qconfig) 
        qat_conv.weight = mod.weight
        qat_conv.bias = mod.bias
        qat_conv.static_padding = mod.static_padding
        return qat_conv


@register_convert_function(QConvStaticSamePaddingConvBn2d)
def convert_qconvstaticsame_convbn(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a Conv2d from FusedModule.
    conv = torch.nn.Conv2d(fused_module.in_channels, fused_module.out_channels, fused_module.kernel_size, 
                           fused_module.stride, fused_module.padding, fused_module.dilation,
                           fused_module.groups, fused_module.bias is not None, fused_module.padding_mode)
    conv.weight = fused_module.weight
    if fused_module.bias is not None:
        conv.bias = fused_module.bias
    fused_conv = nn.utils.fuse_conv_bn_eval(conv.eval(), fused_module.bn)
    # We need nn.qat.conv here to export weight quantize node.
    fused_conv.qconfig = fused_module.qconfig
    fused_conv.static_padding = fused_module.static_padding
    fused_conv = QConvStaticSamePadding2d.from_float(fused_conv)
    # Attach weight fake quantize params.
    fused_conv.weight_fake_quant = fused_module.weight_fake_quant
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    setattr(modules[conv_parent_name], conv_name, fused_conv)
