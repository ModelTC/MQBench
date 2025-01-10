from torch.onnx import register_custom_op_symbolic

# Register symbolic op for torch.quantize_function op.
import functools
from torch.onnx._internal import jit_utils, registration
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=13)
_custom_onnx_symbolic = functools.partial(registration.custom_onnx_symbolic, opset=13)
from torch.onnx import (
    _type_utils,
    symbolic_helper,
    symbolic_opset9 as opset9,
)
import torch._C._onnx as _C_onnx
import torch
@_custom_onnx_symbolic("aten::fake_quantize_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i")
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    # if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
    #     raise errors.SymbolicValueError(
    #         "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
    #         f"Got ({quant_min}, {quant_max})",
    #         inputs,
    #     )
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, quant_min_i = quant_min, quant_max_i = quant_max)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point, quant_min_i = quant_min, quant_max_i = quant_max)

@_custom_onnx_symbolic("aten::fake_quantize_per_channel_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i")
def fake_quantize_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    # if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
    #     raise errors.SymbolicValueError(
    #         "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
    #         f"Got ({quant_min}, {quant_max})",
    #         inputs,
    #     )
    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis, quant_min_i = quant_min, quant_max_i = quant_max)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis, quant_min_i = quant_min, quant_max_i = quant_max)

@_onnx_symbolic("aten::_fake_quantize_learnable_per_tensor_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "f")
def _fake_quantize_learnable_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
    grad_factor=1.0,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    # if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
    #     raise errors.SymbolicValueError(
    #         "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
    #         f"Got ({quant_min}, {quant_max})",
    #         inputs,
    #     )
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, quant_min_i = quant_min, quant_max_i = quant_max)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point, quant_min_i = quant_min, quant_max_i = quant_max)

@_custom_onnx_symbolic("aten::_fake_quantize_learnable_per_channel_affine")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "f")
def _fake_quantize_learnable_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    axis,
    quant_min=-128,
    quant_max=127,
    grad_factor=1.0,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    # if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
    #     raise errors.SymbolicValueError(
    #         "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
    #         f"Got ({quant_min}, {quant_max})",
    #         inputs,
    #     )
    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis, quant_min_i = quant_min, quant_max_i = quant_max)
    if (quant_min, quant_max) == (0, 127):
        quantized = g.op(
            "Clip",
            quantized,
            opset9.unused(g),
            g.op("Constant", value_t=torch.tensor(127, dtype=torch.uint8)),
        )
    return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis, quant_min_i = quant_min, quant_max_i = quant_max)

# def _fake_quantize_learnable_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max, grad_factor):
#     return g.op(x, scale, zero_point, quant_min, quant_max)
#
#
# register_custom_op_symbolic('::_fake_quantize_learnable_per_tensor_affine', _fake_quantize_learnable_per_tensor_affine, 11)
#
#
# def fake_quantize_per_channel_affine(g, x, scale, zero_point, ch_axis, quant_min, quant_max):
#     return g.op("::FixedPerChannelAffine", x, scale, zero_point, ch_axis, quant_min, quant_max)
#
#
# register_custom_op_symbolic('::fake_quantize_per_channel_affine', fake_quantize_per_channel_affine, 11)
#
#
# def fake_quantize_per_tensor_affine(g, x, scale, zero_point, quant_min, quant_max):
#     return g.op("::FixedPerTensorAffine", x, scale, zero_point, quant_min, quant_max)
#
#
# register_custom_op_symbolic('::fake_quantize_per_tensor_affine', fake_quantize_per_tensor_affine, 11)

