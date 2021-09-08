import os.path as osp

import torch
from torch.fx import GraphModule

import mqbench.custom_symbolic_opset  # noqa: F401
import mqbench.fusion_method          # noqa: F401
from mqbench.prepare_by_platform import BackendType
from mqbench.utils import deepcopy_graphmodule
from mqbench.utils.logger import logger
from mqbench.utils.registry import (
    BACKEND_DEPLOY_FUNCTION,
    register_deploy_function,
    FUSED_MODULE_CONVERT_FUNCTION
)
from mqbench.convert_onnx import (
    remove_fakequantize_and_collect_params_nnie,
    remove_fakequantize_and_collect_params
)


@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
def convert_merge_bn(model: GraphModule, **kwargs):
    logger.info("Merge BN for deploy.")
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    for node in nodes:
        if node.op == 'call_module':
            if type(modules[node.target]) in FUSED_MODULE_CONVERT_FUNCTION:
                FUSED_MODULE_CONVERT_FUNCTION[type(modules[node.target])](model, node)


@register_deploy_function(BackendType.Academic)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
def convert_onnx(model: GraphModule, input_shape_dict, dummy_input, onnx_model_path, **kwargs):
    logger.info("Export to onnx.")
    input_names = None
    if dummy_input is None:
        device = next(model.parameters()).device
        dummy_input = {name: torch.rand(shape).to(device) for name, shape in input_shape_dict.items()}
        input_names = list(dummy_input.keys())
        dummy_input = tuple(dummy_input.values())
    torch.onnx.export(model, dummy_input, onnx_model_path,
                      input_names=input_names,
                      opset_version=11,
                      enable_onnx_checker=False)


@register_deploy_function(BackendType.NNIE)
def deploy_qparams_nnie(model: GraphModule, onnx_model_path, **kwargs):
    logger.info("Extract qparams for NNIE.")
    remove_fakequantize_and_collect_params_nnie(onnx_model_path)


@register_deploy_function(BackendType.Tensorrt)
def deploy_qparams_tensorrt(model: GraphModule, onnx_model_path, **kwargs):
    logger.info("Extract qparams for TensorRT.")
    remove_fakequantize_and_collect_params(onnx_model_path, backend='tensorrt')


@register_deploy_function(BackendType.SNPE)
def deploy_qparams_snpe(model: GraphModule, onnx_model_path, **kwargs):
    logger.info("Extract qparams for SNPE.")
    remove_fakequantize_and_collect_params(onnx_model_path, backend='snpe')


@register_deploy_function(BackendType.PPLW8A16)
def deploy_qparams_pplw8a16(model: GraphModule, onnx_model_path, **kwargs):
    logger.info("Extract qparams for PPLW8A16.")
    remove_fakequantize_and_collect_params(onnx_model_path, backend='ppl')


def convert_deploy(model: GraphModule, backend_type: BackendType,
                   input_shape_dict=None, dummy_input=None, output_path='./',
                   model_name='mqbench_model_quantized.onnx'):
    r"""Convert model to onnx model and quantization params depends on backend.

    Args:
        model (GraphModule): GraphModule prepared qat module.
        backend_type (BackendType): specific which backend should be converted to.
        input_shape_dict (dict): keys are model input name(should be forward function
                                 params name, values are list of tensor dims)
        output_path (str, optional): path to save convert results. Defaults to './'.
        model_name (str, optional): name of converted onnx model. Defaults to 'mqbench_model_quantized.onnx'.

    >>> note on input_shape_dict: 
        example: {'input_0': [1, 3, 224, 224]
                'input_1': [1, 3, 112, 112]
                }
        while forward function signature is like:
                def forward(self, input_0, input_1):
                    pass
    """ 
    kwargs = {
        'input_shape_dict': input_shape_dict,
        'dummy_input': dummy_input,
        'output_path': output_path,
        'model_name': model_name,
        'onnx_model_path': osp.join(output_path, model_name)
    }
    deploy_model = deepcopy_graphmodule(model)
    for convert_function in BACKEND_DEPLOY_FUNCTION[backend_type]:
        convert_function(deploy_model, **kwargs)
