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
from mqbench.deploy import (
    remove_fakequantize_and_collect_params_nnie,
    remove_fakequantize_and_collect_params,
    replace_fakequantize_and_collect_params_openvino,
    remove_fakequantize_and_collect_params_tengine,
    remove_fakequantize_and_collect_params_stpu,
    ONNXQLinearPass, ONNXQNNPass
)
import onnx
from onnxsim import simplify
from mqbench.deploy.common import (
    parse_attrs
)
__all__ = ['convert_deploy']
qmin_max_dict = {}
@register_deploy_function(BackendType.STPU)
@register_deploy_function(BackendType.Tengine_u8)
@register_deploy_function(BackendType.PPLCUDA)
@register_deploy_function(BackendType.ONNX_QNN)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
@register_deploy_function(BackendType.Vitis)
@register_deploy_function(BackendType.OPENVINO)
@register_deploy_function(BackendType.QDQ)
def convert_merge_bn(model: GraphModule, **kwargs):
    logger.info("Merge BN for deploy.")
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    for node in nodes:
        if node.op == 'call_module':
            if type(modules[node.target]) in FUSED_MODULE_CONVERT_FUNCTION:
                FUSED_MODULE_CONVERT_FUNCTION[type(modules[node.target])](model, node)


@register_deploy_function(BackendType.STPU)
@register_deploy_function(BackendType.Academic_NLP)
@register_deploy_function(BackendType.Tensorrt_NLP)
@register_deploy_function(BackendType.Tengine_u8)
@register_deploy_function(BackendType.PPLCUDA)
@register_deploy_function(BackendType.ONNX_QNN)
@register_deploy_function(BackendType.Academic)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
@register_deploy_function(BackendType.Vitis)
@register_deploy_function(BackendType.OPENVINO)
@register_deploy_function(BackendType.QDQ)
def convert_onnx(model: GraphModule, input_shape_dict, dummy_input, onnx_model_path, **kwargs):
    logger.info("Export to onnx.")
    output_names = kwargs.get('output_names', [])
    dynamic_axes = kwargs.get('dynamic_axes', {})
    input_names = kwargs.get('input_names', [])
    if dummy_input is None:
        device = next(model.parameters()).device
        dummy_input = {name: torch.rand(shape).to(device) for name, shape in input_shape_dict.items()}
        input_names = list(dummy_input.keys())
        dummy_input = tuple(dummy_input.values())
    # Per-channel QuantizeLinear and DequantizeLinear is supported since opset 13
    opset_version = 13 if kwargs.get('deploy_to_qlinear', False) else 13
    with torch.no_grad():
        # try:
        #     from torch.onnx.utils import ONNXCheckerError
        #     try:
        torch.onnx.export(model, dummy_input, onnx_model_path,
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=opset_version,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          custom_opsets={'' : opset_version})


        #     except ONNXCheckerError:
        #         pass
        # except ImportError:
        #     torch.onnx.export(model, dummy_input, onnx_model_path,
        #                       input_names=input_names,
        #                       output_names=output_names,
        #                       opset_version=opset_version,
        #                       do_constant_folding=True,
        #                       custom_opsets={'' : opset_version},
        #                       enable_onnx_checker=False)
        onnx_model = onnx.load(onnx_model_path)
        graph = onnx_model.graph
        for node in graph.node:
            if len(node.attribute) > 1:
                qparams = parse_attrs(node.attribute)
                if 'quant_max' in qparams:
                    qmin_max_dict[node.name] = (qparams['quant_min'], qparams['quant_max'])
                    new_attributes = []
                    for attr in node.attribute:
                        if attr.name not in ["quant_min", "quant_max"]:
                            new_attributes.append(attr)
                    node.ClearField("attribute")
                    node.attribute.extend(new_attributes)
        onnx.save(onnx_model, onnx_model_path)
        try:
            logger.info("simplify model.")
            onnx_model = onnx.load(onnx_model_path)
            onnx_model_simplified, check = simplify(onnx_model)
            onnx.save(onnx_model_simplified, onnx_model_path)
        except Exception as e:
            logger.info("simplify model fail.")
        # onnx.checker.check_model(onnx_model_simplified)
        # import onnxruntime as ort
        # session = ort.InferenceSession(onnx_model_path)

@register_deploy_function(BackendType.Tensorrt)
def convert_onnx_qlinear(model: GraphModule, onnx_model_path, model_name, **kwargs):
    if kwargs.get('deploy_to_qlinear', False):
        logger.info("Convert to ONNX QLinear.")
        ONNXQLinearPass(onnx_model_path).run()


@register_deploy_function(BackendType.NNIE)
def deploy_qparams_nnie(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for NNIE.")
    remove_fakequantize_and_collect_params_nnie(onnx_model_path, model_name)


@register_deploy_function(BackendType.OPENVINO)
def deploy_qparams_openvino(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for OPENVINO.")
    replace_fakequantize_and_collect_params_openvino(onnx_model_path, model_name, qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.Tensorrt)
def deploy_qparams_tensorrt(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for TensorRT.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='tensorrt', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.Vitis)
def deploy_qparams_vitis(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Vitis-DPU.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='vitis', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.SNPE)
def deploy_qparams_snpe(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for SNPE.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='snpe', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.PPLW8A16)
def deploy_qparams_pplw8a16(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for PPLW8A16.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='ppl', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.ONNX_QNN)
def deploy_qparams_tvm(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Convert to ONNX QNN.")
    ONNXQNNPass(onnx_model_path).run(model_name, qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.PPLCUDA)
def deploy_qparams_ppl_cuda(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for PPL-CUDA.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='ppl-cuda', qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.Tengine_u8)
def deploy_qparams_tengine(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Tengine.")
    remove_fakequantize_and_collect_params_tengine(onnx_model_path, model_name, qmin_max_dict = qmin_max_dict)


@register_deploy_function(BackendType.STPU)
def deploy_qparams_stpu(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for STPU.")
    remove_fakequantize_and_collect_params_stpu(onnx_model_path, model_name, qmin_max_dict = qmin_max_dict)


def  convert_deploy(model: GraphModule, backend_type: BackendType,
                   input_shape_dict=None, dummy_input=None, output_path='./',
                   model_name='mqbench_qmodel', deploy_to_qlinear=False, **extra_kwargs):
    r"""Convert model to onnx model and quantization params depends on backend.

    Args:
        model (GraphModule): GraphModule prepared qat module.
        backend_type (BackendType): specific which backend should be converted to.
        input_shape_dict (dict): keys are model input name(should be forward function
                                 params name, values are list of tensor dims)
        output_path (str, optional): path to save convert results. Defaults to './'.
        model_name (str, optional): name of converted onnx model. Defaults to 'mqbench_qmodel'.

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
        'onnx_model_path': osp.join(output_path, '{}.onnx'.format(model_name)),
        'deploy_to_qlinear': deploy_to_qlinear,
    }
    # kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    for convert_function in BACKEND_DEPLOY_FUNCTION[backend_type]:
        convert_function(deploy_model, **kwargs)
