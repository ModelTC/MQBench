import torch
import torch.nn as nn
# from torch.quantization.fx.fusion_patterns import ConvBNReLUFusion, ModuleReLUFusion
from torch.quantization.fx.quantization_types import QuantizerCls
from torch.fx.graph import Node
from collections import namedtuple

import mqbench.nn as qnn
import mqbench.nn.intrinsic as qnni
import mqbench.nn.intrinsic.qat as qnniqat
from mqbench.utils.fusion import fuse_deconv_bn_eval
from mqbench.nn.modules import FrozenBatchNorm2d
from torch.ao.quantization.fx.fuse_handler import DefaultFuseHandler
from torch.ao.quantization.backend_config import (
    BackendPatternConfig,
    DTypeConfig,
    DTypeWithConstraints,
    ObservationType,
)
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized.reference as nnqr
import torch.nn.functional as F




def _get_custom_conv_configs(dtype_configs):
    """
    Return all configs related to conv modules and ops.
    """
    conv_configs = []
    observation_type = ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT
    # 1 conv transpose + bn + relu
    conv_configs.append(
        BackendPatternConfig((nn.ConvTranspose2d, nn.BatchNorm2d, nn.ReLU))
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(fuse_deconv_bn_relu)
        .set_fused_module(qnni.ConvTransposeBnReLU2d)
    )

    conv_configs.append(
        BackendPatternConfig(qnni.ConvTransposeBnReLU2d)
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_observation_type(observation_type)
        .set_root_module(nn.ConvTranspose2d)
        .set_reference_quantized_module(nnqr.ConvTranspose2d)
        .set_qat_module(qnniqat.ConvTransposeBnReLU2d)
    )
    # 2 conv transpose + bn
    conv_configs.append(
        BackendPatternConfig((nn.ConvTranspose2d, nn.BatchNorm2d))
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(fuse_deconv_bn)
        .set_fused_module(qnni.ConvTransposeBn2d)
    )
    conv_configs.append(
        BackendPatternConfig(qnni.ConvTransposeBn2d)
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_observation_type(observation_type)
        .set_root_module(nn.ConvTranspose2d)
        .set_reference_quantized_module(nnqr.ConvTranspose2d)
        .set_qat_module(qnniqat.ConvTransposeBn2d)
    )
    # 3 conv transpose
    conv_configs.append(
        BackendPatternConfig(nn.ConvTranspose2d)
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_observation_type(observation_type)
        .set_root_module(nn.ConvTranspose2d)
        .set_reference_quantized_module(nnqr.ConvTranspose2d)
        .set_qat_module(qnn.qat.ConvTranspose2d)
    )
    # 4 linear bn
    conv_configs.append(
        BackendPatternConfig((nn.Linear, nn.BatchNorm1d))
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_fuser_method(fuse_linear_bn)
        .set_fused_module(qnni.LinearBn1d)
    )
    conv_configs.append(
        BackendPatternConfig(qnni.LinearBn1d)
        .set_dtype_configs(dtype_configs)  # noqa: E131
        .set_observation_type(observation_type)
        .set_root_module(nn.Linear)
        .set_reference_quantized_module(nnqr.Linear)
        .set_qat_module(qnniqat.LinearBn1d)
    )

    return conv_configs


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
    assert linear.training == bn.training, \
        "Linear and BN both must be in the same mode (train or eval)."

    if linear.training:
        assert bn.affine, 'Only support fusing BatchNorm1d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm1d with tracking_running_stats set to True'
        return qnn.intrinsic.LinearBn1d(linear, bn)
    else:
        return nn.utils.fusion.fuse_linear_bn_eval(linear, bn)


def fuse_deconv_bn(is_qat, deconv, bn):
    assert deconv.training == bn.training, \
        'DeConv and BN must be in the same mode (train or eval)'

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_bn_relu(is_qat, deconv, bn, relu):
    assert deconv.training == bn.training == relu.training, \
        "DeConv and BN both must be in the same mode (train or eval)."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeBnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)


def fuse_conv_freezebn(is_qat, conv, bn):
    assert bn.training is False, "Freezebn must be eval."

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvFreezebn2d(conv, bn)
    else:
        return nn.utils.fuse_conv_bn_eval(conv, bn)


def fuse_conv_freezebn_relu(is_qat, conv, bn, relu):
    assert conv.training == relu.training and bn.training is False, \
        "Conv and relu both must be in the same mode (train or eval) and bn must be eval."

    if conv.training:
        assert bn.num_features == conv.out_channels, 'Output channel of Conv must match num_features of BatchNorm'
        assert bn.affine, 'Only support fusing BatchNorm with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm with tracking_running_stats set to True'
        return qnni.ConvFreezebnReLU2d(conv, bn, relu)
    else:
        fused_conv = nn.utils.fuse_conv_bn_eval(conv, bn)
        return nn.intrinsic.ConvReLU2d(fused_conv, relu)


def fuse_deconv_freezebn(is_qat, deconv, bn):
    assert bn.training is False, "Freezebn must be eval."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeFreezebn2d(deconv, bn)
    else:
        return fuse_deconv_bn_eval(deconv, bn)


def fuse_deconv_freezebn_relu(is_qat, deconv, bn, relu):
    assert deconv.training == relu.training and bn.training is False, \
        "Conv and relu both must be in the same mode (train or eval) and bn must be eval."

    if deconv.training:
        assert bn.num_features == deconv.out_channels, 'Output channel of ConvTranspose2d must match num_features of BatchNorm2d'
        assert bn.affine, 'Only support fusing BatchNorm2d with affine set to True'
        assert bn.track_running_stats, 'Only support fusing BatchNorm2d with tracking_running_stats set to True'
        return qnni.ConvTransposeFreezebnReLU2d(deconv, bn, relu)
    else:
        return qnni.ConvTransposeReLU2d(fuse_deconv_bn_eval(deconv, bn), relu)



