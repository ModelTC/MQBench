import torch
import torch.nn.intrinsic.qat as nniqat
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_linear_bn_eval

# turn foo.bar -> ['foo', 'bar']
def _parent_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    else:
        return r[0], r[1]

import sophgo_mq.nn.intrinsic as qnni
import sophgo_mq.nn.intrinsic.qat as qnniqat
import sophgo_mq.nn.qat as qnnqat
from sophgo_mq.utils.registry import register_convert_function
from sophgo_mq.fuser_method_mappings import fuse_deconv_bn_eval
from sophgo_mq.quantization.default_bias_fake_quant import bias_fake_quantizer


@register_convert_function(qnniqat.Linear_sophgo)
def convert_qnniqat_linear(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a Linear from FusedModule.
    linear = torch.nn.Linear(fused_module.in_features, fused_module.out_features, fused_module.bias is not None)
    linear.weight = fused_module.weight
    if fused_module.bias is not None:
        linear.bias = fused_module.bias
    # We need nn.qat.linear here to export weight quantize node.
    linear.qconfig = fused_module.qconfig
    linear = torch.nn.qat.Linear.from_float(linear)
    # Attach weight fake quantize params.
    linear.weight_fake_quant = fused_module.weight_fake_quant
    linear_parent_name, linear_name = _parent_name(fused_node.target)
    setattr(modules[linear_parent_name], linear_name, linear)

@register_convert_function(qnnqat.Conv2d_sophgo)
def convert_qnnqat_conv2d(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a Conv2d from FusedModule.
    conv = torch.nn.Conv2d(fused_module.in_channels, fused_module.out_channels, fused_module.kernel_size, 
                           fused_module.stride, fused_module.padding, fused_module.dilation,
                           fused_module.groups, fused_module.bias is not None, fused_module.padding_mode)
    conv.weight = fused_module.weight
    if fused_module.bias is not None:
        conv.bias = fused_module.bias
    # We need nn.qat.conv here to export weight quantize node.
    conv.qconfig = fused_module.qconfig
    conv = torch.nn.qat.Conv2d.from_float(conv)
    # Attach weight fake quantize params.
    conv.weight_fake_quant = fused_module.weight_fake_quant
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    setattr(modules[conv_parent_name], conv_name, conv)

@register_convert_function(qnnqat.ConvTranspose2d_sophgo)
def convert_qnnqat_deconv2d(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    deconv = torch.nn.ConvTranspose2d(fused_module.in_channels, fused_module.out_channels, fused_module.kernel_size, 
                                      stride=fused_module.stride, padding=fused_module.padding, output_padding=fused_module.output_padding,
                                      groups=fused_module.groups, bias=fused_module.bias is not None, 
                                      dilation=fused_module.dilation,
                                      padding_mode=fused_module.padding_mode) 
    deconv.weight = fused_module.weight
    if fused_module.bias is not None:
        deconv.bias = fused_module.bias
    fused_deconv = fuse_deconv_bn_eval(deconv.eval(), fused_module.bn)
    fused_deconv.qconfig = fused_module.qconfig
    fused_deconv = qnnqat.ConvTranspose2d.from_float(fused_deconv)
    fused_deconv.weight_fake_quant = fused_module.weight_fake_quant
    deconv_parent_name, deconv_name = _parent_name(fused_node.target)
    setattr(modules[deconv_parent_name], deconv_name, fused_deconv)

@register_convert_function(qnniqat.LinearReLU_sophgo)
def linearert_qnniqat_linearrelu(model, fused_node):
    convert_qnniqat_linear(model, fused_node)
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # We need to Insert Relu after Merged linear.
    linear_parent_name, linear_name = _parent_name(fused_node.target)
    relu_name = 'relu'
    # Maybe has another name, but we cannot know for now.
    if not hasattr(modules[linear_parent_name], relu_name):
        setattr(modules[linear_parent_name], relu_name, 
                torch.nn.ReLU(inplace=True).train(fused_module.training))
    # Update modules.
    modules = dict(model.named_modules())
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if linear_parent_name == "" else "{}.{}".format(linear_parent_name, relu_name)
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {})
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()

@register_convert_function(qnni.LinearBn1d)
def convert_qnni_linearbn(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    fused_linear = fuse_linear_bn_eval(fused_module[0], fused_module[1])
    linear_parent_name, linear_name = _parent_name(fused_node.target)
    setattr(modules[linear_parent_name], linear_name, fused_linear)


@register_convert_function(qnniqat.LinearBn1d)
@register_convert_function(qnniqat.LinearBn1d_sophgo)
def convert_qnniqat_linearbn(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a Linear from FusedModule.
    linear = torch.nn.Linear(fused_module.in_features, fused_module.out_features, fused_module.bias is not None)
    linear.weight = fused_module.weight
    if fused_module.bias is not None:
        linear.bias = fused_module.bias
    # Merge Linear + BN
    fused_linear = fuse_linear_bn_eval(linear.eval(), fused_module.bn)
    # We need nn.qat.linear here to export weight quantize node.
    fused_linear.qconfig = fused_module.qconfig
    fused_linear = torch.nn.qat.Linear.from_float(fused_linear)
    # Attach weight fake quantize params.
    fused_linear.weight_fake_quant = fused_module.weight_fake_quant
    linear_parent_name, linear_name = _parent_name(fused_node.target)
    setattr(modules[linear_parent_name], linear_name, fused_linear)


@register_convert_function(qnniqat.ConvFreezebn2d)
@register_convert_function(nniqat.ConvBn2d)
@register_convert_function(nniqat.ConvBn3d)
@register_convert_function(qnniqat.ConvBn2d_sophgo)
def convert_nniqat_convbn(model, fused_node):
    """nniqat.ConvBn2d ----> nn.Conv2d ----> nniqat.Conv2d
    """
    fused_module_class_map = {
        qnniqat.ConvFreezebn2d: torch.nn.Conv2d,
        qnniqat.ConvFreezebnReLU2d: torch.nn.Conv2d,
        nniqat.ConvBn2d: torch.nn.Conv2d,
        nniqat.ConvBnReLU2d: torch.nn.Conv2d,
        nniqat.ConvBn3d: torch.nn.Conv3d,
        nniqat.ConvBnReLU3d: torch.nn.Conv3d,
        qnniqat.ConvBn2d_sophgo: torch.nn.Conv2d,
        qnniqat.ConvBnReLU2d_sophgo: torch.nn.Conv2d,
    }
    fused_qat_module_class_map = {
        torch.nn.Conv2d: torch.nn.qat.Conv2d,
        torch.nn.Conv3d: torch.nn.qat.Conv3d,
    }
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a Conv2d from FusedModule.
    conv = fused_module_class_map[type(fused_module)](fused_module.in_channels, fused_module.out_channels,
                                                      fused_module.kernel_size, fused_module.stride,
                                                      fused_module.padding, fused_module.dilation,
                                                      fused_module.groups, fused_module.bias is not None,
                                                      fused_module.padding_mode)
    conv.weight = fused_module.weight
    if fused_module.bias is not None:
        conv.bias = fused_module.bias
    fused_conv = fuse_conv_bn_eval(conv.eval(), fused_module.bn)
    # We need nn.qat.conv here to export weight quantize node.
    fused_conv.qconfig = fused_module.qconfig
    fused_conv = fused_qat_module_class_map[type(conv)].from_float(fused_conv)
    # Attach weight fake quantize params.
    fused_conv.weight_fake_quant = fused_module.weight_fake_quant
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    setattr(modules[conv_parent_name], conv_name, fused_conv)


@register_convert_function(qnniqat.ConvFreezebnReLU2d)
@register_convert_function(nniqat.ConvBnReLU2d)
@register_convert_function(nniqat.ConvBnReLU3d)
@register_convert_function(qnniqat.ConvBnReLU2d_sophgo)
def convert_nniqat_convbnrelu(model, fused_node):    
    convert_nniqat_convbn(model, fused_node)
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # We need to Insert Relu after Merged conv.
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    relu_name = 'relu'
    # Maybe has another name, but we cannot know for now.
    if not hasattr(modules[conv_parent_name], relu_name):
        setattr(modules[conv_parent_name], relu_name, 
                torch.nn.ReLU(inplace=True).train(fused_module.training))
    # Update modules.
    modules = dict(model.named_modules())
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if conv_parent_name == "" else "{}.{}".format(conv_parent_name, relu_name)
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {})
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()

@register_convert_function(qnniqat.ConvReLU2d_sophgo)
def convert_nniqat_convrelu(model, fused_node):    
    convert_qnnqat_conv2d(model, fused_node)
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # We need to Insert Relu after Merged conv.
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    relu_name = 'relu'
    # Maybe has another name, but we cannot know for now.
    if not hasattr(modules[conv_parent_name], relu_name):
        setattr(modules[conv_parent_name], relu_name, 
                torch.nn.ReLU(inplace=True).train(fused_module.training))
    # Update modules.
    modules = dict(model.named_modules())
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if conv_parent_name == "" else "{}.{}".format(conv_parent_name, relu_name)
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {})
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()

@register_convert_function(qnni.ConvTransposeFreezebn2d)
@register_convert_function(qnni.ConvTransposeBn2d)
def convert_qnni_deconvbn(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    fused_module_deconv = fused_module[0]
    fused_module_bn = fused_module[1]
    # Create a ConvTranspose2d from FusedModule.
    deconv = torch.nn.ConvTranspose2d(fused_module_deconv.in_channels, fused_module_deconv.out_channels, fused_module_deconv.kernel_size, 
                                      stride=fused_module_deconv.stride, padding=fused_module_deconv.padding, output_padding=fused_module_deconv.output_padding,
                                      groups=fused_module_deconv.groups, bias=fused_module_deconv.bias is not None, 
                                      dilation=fused_module_deconv.dilation,
                                      padding_mode=fused_module_deconv.padding_mode) 
    deconv.weight = fused_module_deconv.weight
    if fused_module_deconv.bias is not None:
        deconv.bias = fused_module_deconv.bias
    fused_deconv = fuse_deconv_bn_eval(deconv.eval(), fused_module_bn)
    deconv_parent_name, deconv_name = _parent_name(fused_node.target)
    setattr(modules[deconv_parent_name], deconv_name, fused_deconv)


@register_convert_function(qnniqat.ConvTransposeFreezebn2d)
@register_convert_function(qnniqat.ConvTransposeBn2d)
@register_convert_function(qnniqat.ConvTransposeBn2d_sophgo)
def convert_qnniqat_deconvbn(model, fused_node):
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a ConvTranspose2d from FusedModule.
    deconv = torch.nn.ConvTranspose2d(fused_module.in_channels, fused_module.out_channels, fused_module.kernel_size, 
                                      stride=fused_module.stride, padding=fused_module.padding, output_padding=fused_module.output_padding,
                                      groups=fused_module.groups, bias=fused_module.bias is not None, 
                                      dilation=fused_module.dilation,
                                      padding_mode=fused_module.padding_mode) 
    deconv.weight = fused_module.weight
    if fused_module.bias is not None:
        deconv.bias = fused_module.bias
    fused_deconv = fuse_deconv_bn_eval(deconv.eval(), fused_module.bn)
    # We need nn.qat.conv here to export weight quantize node.
    fused_deconv.qconfig = fused_module.qconfig
    fused_deconv = qnnqat.ConvTranspose2d.from_float(fused_deconv)
    # Attach weight fake quantize params.
    fused_deconv.weight_fake_quant = fused_module.weight_fake_quant
    deconv_parent_name, deconv_name = _parent_name(fused_node.target)
    setattr(modules[deconv_parent_name], deconv_name, fused_deconv)


@register_convert_function(qnni.ConvTransposeFreezebnReLU2d)
@register_convert_function(qnni.ConvTransposeBnReLU2d)
@register_convert_function(qnniqat.ConvTransposeBnReLU2d_sophgo)
def convert_qnni_deconvbnrelu(model, fused_node):
    convert_qnni_deconvbn(model, fused_node)
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    deconv_parent_name, deconv_name = _parent_name(fused_node.target)
    relu_name = 'relu'
    if not hasattr(modules[deconv_parent_name], relu_name):
        setattr(modules[deconv_parent_name], relu_name, torch.nn.ReLU(inplace=True).train(fused_module.training))
    # Update modules.
    modules = dict(model.named_modules())
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if deconv_parent_name == "" else "{}.{}".format(deconv_parent_name, relu_name)
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {})
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()


@register_convert_function(qnniqat.ConvTransposeFreezebnReLU2d)
@register_convert_function(qnniqat.ConvTransposeBnReLU2d)
@register_convert_function(qnniqat.ConvTransposeBnReLU2d_sophgo)
def convert_qnniqat_deconvbnrelu(model, fused_node):
    convert_qnniqat_deconvbn(model, fused_node)
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    deconv_parent_name, deconv_name = _parent_name(fused_node.target)
    relu_name = 'relu'
    if not hasattr(modules[deconv_parent_name], relu_name):
        setattr(modules[deconv_parent_name], relu_name, torch.nn.ReLU(inplace=True).train(fused_module.training))
    # Update modules.
    modules = dict(model.named_modules())
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if deconv_parent_name == "" else "{}.{}".format(deconv_parent_name, relu_name)
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {})
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()


@register_convert_function(qnniqat.ConvBn2d)
def convert_qnniqat_convbn(model, fused_node):
    """sophgo_mq.nn.intrinsic.qat module add bias quant.
    That is the difference between torch.nn.intrinsic.qat module.
    """
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # Create a Conv2d from FusedModule.
    conv = torch.nn.Conv2d(fused_module.in_channels, fused_module.out_channels, fused_module.kernel_size, 
                           fused_module.stride, fused_module.padding, fused_module.dilation,
                           fused_module.groups, fused_module.bias is not None, fused_module.padding_mode)
    conv.weight = fused_module.weight
    if fused_module.bias is not None:
        conv.bias = fused_module.bias
    fused_conv = fuse_conv_bn_eval(conv.eval(), fused_module.bn)
    # We need nn.qat.conv here to export weight quantize node.
    fused_conv.qconfig = fused_module.qconfig
    fused_conv = qnnqat.Conv2d.from_float(fused_conv)
    # Attach weight fake quantize params.
    fused_conv.weight_fake_quant = fused_module.weight_fake_quant
    if hasattr(fused_module, 'bias_fake_quant'):
        fused_conv.bias_fake_quant = fused_module.bias_fake_quant
    else:
        fused_conv.bias_fake_quant = bias_fake_quantizer()
        fused_conv.bias_fake_quant.set_quant_type('param')
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    setattr(modules[conv_parent_name], conv_name, fused_conv)


@register_convert_function(qnniqat.ConvBnReLU2d)
def convert_qnniqat_convbnrelu(model, fused_node):
    """sophgo_mq.nn.intrinsic.qat module add bias quant.
    That is the difference between torch.nn.intrinsic.qat module.
    """
    convert_qnniqat_convbn(model, fused_node)
    modules = dict(model.named_modules())
    fused_module = modules[fused_node.target]
    # We need to Insert Relu after Merged conv.
    conv_parent_name, conv_name = _parent_name(fused_node.target)
    relu_name = 'relu'
    # Maybe has another name, but we cannot know for now.
    if not hasattr(modules[conv_parent_name], relu_name):
        setattr(modules[conv_parent_name], relu_name, 
                torch.nn.ReLU(inplace=True).train(fused_module.training))
    # Update modules.
    modules = dict(model.named_modules())
    graph = model.graph
    nodes = list(model.graph.nodes)
    with graph.inserting_after(fused_node):
        relu_node_name = relu_name if conv_parent_name == "" else "{}.{}".format(conv_parent_name, relu_name)
        assert relu_node_name in modules and isinstance(modules[relu_node_name], torch.nn.ReLU)
        inserted_node = graph.create_node("call_module", relu_node_name, (fused_node,), {}, name = fused_node.name)
        for _node in nodes:
            for i, _arg in enumerate(_node.args):
                if _arg == fused_node:
                    _tmp = list(_node.args)
                    _tmp[i] = inserted_node
                    _node.args = tuple(_tmp)
    model.recompile()
    model.graph.lint()