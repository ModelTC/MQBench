import torch
import torch.nn as nn
from torch.quantization.fx.fusion_patterns import ConvBNReLUFusion, ModuleReLUFusion

import mqbench.nn as qnn
import mqbench.nn.intrinsic as qnni
import mqbench.nn.intrinsic.qat as qnniqat
from mqbench.utils.fusion import fuse_deconv_bn_eval


def fuse_linear_bn(linear, bn):
    r"""Given the linear and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type Linear
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Linear(10, 20)
        >>> b1 = nn.BatchNorm1d(20)
        >>> m2 = fuse_linear_bn(m1, b1)
    """
    assert(linear.training == bn.training),\
        "Linear and BN both must be in the same mode (train or eval)."

    if linear.training:
        assert bn.affine, 'Only support fusing BatchNorm1d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm1d with tracking_running_stats set to True'
        return qnn.intrinsic.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_deconv_bn(deconv, bn):
    assert(deconv.training == bn.training),\
        'DeConv and BN must be in the same mode (train or eval)'

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_bn_relu(deconv, bn, relu):
    assert(deconv.training == bn.training == relu.training),\
        "DeConv and BN both must be in the same mode (train or eval)."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)


fuse_custom_config_dict = {
    "additional_fuser_method_mapping": {
        (torch.nn.Linear, torch.nn.BatchNorm1d): fuse_linear_bn,
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d): fuse_deconv_bn,
        (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU): fuse_deconv_bn_relu,
        (torch.nn.ConvTranspose2d, torch.nn.ReLU): qnni.ConvTransposeReLU2d,
    },
    "additional_fusion_pattern": {
        (torch.nn.BatchNorm1d, torch.nn.Linear):
        ConvBNReLUFusion,
        (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, torch.nn.ConvTranspose2d):
        ConvBNReLUFusion,
        (torch.nn.functional.relu, (torch.nn.BatchNorm2d, torch.nn.ConvTranspose2d)):
        ConvBNReLUFusion,
    },
    "additional_qat_module_mappings": {
        qnni.LinearBn1d: qnniqat.LinearBn1d,
        qnni.ConvTransposeBn2d: qnniqat.ConvTransposeBn2d,
        qnni.ConvTransposeReLU2d: qnniqat.ConvTransposeReLU2d,
        qnni.ConvTransposeBnReLU2d: qnniqat.ConvTransposeBnReLU2d
    },
}


def _sort_fusion_patterns(pats):
    keys = []
    for key in pats.keys():
        if pats[key] is ModuleReLUFusion:
            keys.append(key)
    for key in keys:
        pats.move_to_end(key)


# Sinse additional_fuser_method_mapping will not be set because fuser.py:54
# do not pass this dict.
from torch.quantization.fuser_method_mappings import DEFAULT_OP_LIST_TO_FUSER_METHOD
from torch.quantization.fx.pattern_utils import DEFAULT_FUSION_PATTERNS
from torch.quantization.quantization_mappings import DEFAULT_QAT_MODULE_MAPPINGS

DEFAULT_OP_LIST_TO_FUSER_METHOD.update(
    fuse_custom_config_dict['additional_fuser_method_mapping'])
DEFAULT_FUSION_PATTERNS.update(
    fuse_custom_config_dict['additional_fusion_pattern'])
# Make longer matched pattern prior.
# i.e. Conv + BN + Relu should match ConvBnRelu before BNRelu.
# Any thing registered in class ConvBNReLUFusion should be
# proir than class ModuleReLUFusion.
_sort_fusion_patterns(DEFAULT_FUSION_PATTERNS)
DEFAULT_QAT_MODULE_MAPPINGS.update(
    fuse_custom_config_dict['additional_qat_module_mappings'])