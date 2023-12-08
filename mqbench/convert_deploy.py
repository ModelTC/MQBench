import json
import os.path as osp
import os
import torch
from torch.fx import GraphModule
import onnx
from onnxsim import simplify
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
    remove_fakequantize_and_collect_params_flt,
    replace_fakequantize_and_collect_params_openvino,
    remove_fakequantize_and_collect_params_sophgo,
    # remove_fakequantize_and_collect_params_academic,
    ONNXQLinearPass, ONNXQNNPass
)

__all__ = ['convert_deploy']

@register_deploy_function(BackendType.PPLCUDA)
@register_deploy_function(BackendType.ONNX_QNN)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
@register_deploy_function(BackendType.Vitis)
@register_deploy_function(BackendType.OPENVINO)
@register_deploy_function(BackendType.Sophgo_TPU)
def convert_merge_bn(model: GraphModule, **kwargs):
    # print('wlog before convert_merge_bn, model.named_modules:', dict(model.named_modules())[''])
    # print('wlog before convert_merge_bn, model.graph:', model.graph)
    logger.info("Merge BN for deploy.")
    nodes = list(model.graph.nodes)
    modules = dict(model.named_modules())
    for node in nodes:
        if node.op == 'call_module':
            if node.target in modules and type(modules[node.target]) in FUSED_MODULE_CONVERT_FUNCTION:
                FUSED_MODULE_CONVERT_FUNCTION[type(modules[node.target])](model, node)
    # print('wlog after convert_merge_bn, model.named_modules:', dict(model.named_modules())[''])
    # print('wlog after convert_merge_bn, model.graph:', model.graph)

@register_deploy_function(BackendType.Academic_NLP)
@register_deploy_function(BackendType.Tensorrt_NLP)
@register_deploy_function(BackendType.PPLCUDA)
@register_deploy_function(BackendType.ONNX_QNN)
@register_deploy_function(BackendType.Academic)
@register_deploy_function(BackendType.SNPE)
@register_deploy_function(BackendType.PPLW8A16)
@register_deploy_function(BackendType.Tensorrt)
@register_deploy_function(BackendType.NNIE)
@register_deploy_function(BackendType.Vitis)
@register_deploy_function(BackendType.OPENVINO)
@register_deploy_function(BackendType.Sophgo_TPU)
def convert_onnx(model: GraphModule, input_shape_dict, dummy_input, onnx_model_path, **kwargs):
    pt_file_name = onnx_model_path.split('.')
    pt_file_name[-1] = 'pt'
    #torch.save(model, '.'.join(pt_file_name))
    logger.info("Export to onnx, onnx_model_path:{}".format(onnx_model_path))
    model = model.cpu()
    output_names = kwargs.get('output_names', [])
    dynamic_axes = kwargs.get('dynamic_axes', {})
    input_names = kwargs.get('input_names', [])
    if dummy_input is None:
        device = next(model.parameters()).device
        dummy_input = {name: torch.rand(shape).to(device) for name, shape in input_shape_dict.items()}
        input_names = list(dummy_input.keys())
        dummy_input = tuple(dummy_input.values())
    # Per-channel QuantizeLinear and DequantizeLinear is supported since opset 13
    opset_version = 13 if kwargs.get('deploy_to_qlinear', False) else 11
    # opset_version = 18

    # open all fake quant node to export
    if isinstance(model, torch.fx.graph_module.GraphModule):
        print(">>>>> print graphmodule before export", model)
        # print(">>>>> print graph before export")
        # model.graph.print_tabular()
        for name, submodule in model.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                if submodule.only_enable_observer == True:
                    submodule.only_enable_observer = False
                submodule.disable_observer()
                submodule.enable_fake_quant()

    with torch.no_grad():
        try:
            from torch.onnx.utils import ONNXCheckerError
            try:
                torch.onnx.export(model, dummy_input, onnx_model_path,
                                  input_names=input_names,
                                  output_names=output_names,
                                  opset_version=opset_version,
                                  dynamic_axes=dynamic_axes,
                                  do_constant_folding=True,
                                  custom_opsets={'MQBench_custom' : opset_version})
            except ONNXCheckerError:
                pass
        except ImportError:
            ### torch 1.13 and 2.0.1
            torch.onnx.export(model, dummy_input, onnx_model_path,
                                input_names=input_names,
                                output_names=output_names,
                                opset_version=opset_version,
                                do_constant_folding=True,
                                custom_opsets={'MQBench_custom' : opset_version})
            tmp_model = onnx.load(onnx_model_path)
            simplified_model, check = simplify(tmp_model)
            onnx.save_model(simplified_model, onnx_model_path)

    model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)
    model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
    os.system(f"rm -f {onnx_model_path}")
    onnx.save(model_onnx, onnx_model_path)

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
    replace_fakequantize_and_collect_params_openvino(onnx_model_path, model_name)


@register_deploy_function(BackendType.Tensorrt)
def deploy_qparams_tensorrt(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for TensorRT.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='tensorrt')

def export_qtable(context_filename, model_name, output_path, quant_mode):
        print("导出qtable")
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[quant_mode]
        file_h.close()
        q_table = osp.join(output_path, '{}_q_table_from_mqbench_{}'.format(model_name, quant_mode))
        with open(q_table, 'w') as f:
            f.write("# qtable from MQBench\n")
            f.write("# op_name  type\n")
            for name,value in blob_range.items():
                dtype = 'INT8'
                if value['bit'] == 4:
                    dtype = 'INT4'
                elif value['type'] == 'uint':
                    dtype = 'UINT8'
                if 'threshold' in value:
                    f.write("{} {}\n".format(name[:-2], dtype))
                else:
                    f.write("{} {}\n".format(name, dtype))

#2023.7.27修改
@register_deploy_function(BackendType.Academic_NLP)
def deploy_qparams_Academic_NLP(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Academic_NLP.")
    # 添加float浮点类型deploy&cali-table输出（目前使用int的onnx）
    nodes = list(model.graph.nodes)
    test_nodes = []
    for node in nodes:
        if "post_act_fake_quantizer" in str(node):
            test_nodes.append(str(node))
    modules = dict(model.named_modules())
    if len(test_nodes) > 2:
        num = len(test_nodes) // 2
    item = modules.get(test_nodes[num])
    if isinstance(item, mqbench.fake_quantize.e4m3.E4M3FakeQuantize):
        mode = "E4M3"
    elif isinstance(item, mqbench.fake_quantize.e5m2.E5M2FakeQuantize):
        mode = "E5M2"
    else:
        mode = "INT"
    # 根据Fake Quantizer的选择判断mode，如果是浮点的mode则进入deploy_float，反之则进入deploy_linear
    # mode = "E4M3" 目前尚未实现fp8的onnx，因此需要暂时使用int的onnx，这里手动将mode定义，进入deploy_float模块，之后实现浮点的onnx后可以删除此段代码
    cali_mode = "Academic_NLP"
    if mode in ["E4M3", "E5M2"]:
        remove_fakequantize_and_collect_params_flt(onnx_model_path, model_name, backend='Academic_NLP') # 修改flt deploy or linear deploy
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[cali_mode+"_Float"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_float_cali_table_from_mqbench_Academic_NLP'.format(model_name))
        fp8_header = "mqbench-fp8"
        with open(cali_table, 'w') as f:
            f.write(f"# work_mode:{fp8_header} #Automatically generated, do not modify, work_mode choice:[E4ME_RNE, E5M2_RNE]\n")
            f.write("#       op_name        threshold        min        max\n")
            weight_scale_fp8 = []
            #fp8_th = []
            for name,value in blob_range.items():
                if 'threshold' in value:
                    tmpstr = "{}     {:.7f}     {:.7f}     {:.7f}\n".format(name[:-2], value['threshold'], value['min'], value['max'])
                    if name.endswith('_fp8'):
                        f.write(tmpstr)
                    else:
                        f.write("{}     {:.7f}     {:.7f}     {:.7f}\n".format(name, value['threshold'], value['min'], value['max']))
                else:
                    tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join([str(i) for i in value['step']]), 
                            len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                    if name.endswith('_weight_fp8') or name.endswith('_bias_fp8'):
                        weight_scale_fp8.append(tmpstr)
                    else:
                        f.write(tmpstr)
            # f.write('#fp8_th\n')
            # for i in fp8_th:
            #     f.write(i)
            f.write('#weight_scale_fp8\n')
            for i in weight_scale_fp8:
                f.write(i)
        cali_mode_new = cali_mode + '_Float'
        export_qtable(context_filename, model_name, output_path, cali_mode_new)
    else:
        remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='Academic_NLP')
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())["Academic_NLP"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_cali_table_from_mqbench_Academic_NLP'.format(model_name))
        work_mode = kwargs.get('work_mode', 'QAT_all_int8')
        if work_mode not in  ['QAT_all_int8', 'int4_and_int8_mix', 'int4_and_int8_mix_no_fc']:
            print('QAT_all_int8 not in [QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc],set to QAT_all_int8')
            work_mode = 'QAT_all_int8'
        with open(cali_table, 'w') as f:
            f.write(f"# work_mode:{work_mode} #Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n")
            f.write("# op_name    threshold    min    max\n")
            weight_scale = []
            int4_th = []
            for name,value in blob_range.items():
                if 'threshold' in value:
                    tmpstr = "{} {:.7f} {:.7f} {:.7f}\n".format(name[:-2], value['threshold'], value['min'], value['max'])
                    if name.endswith('_4'):
                        int4_th.append(tmpstr)
                    elif name.endswith('_8'):
                        f.write(tmpstr)
                    else:
                        f.write("{} {:.7f} {:.7f} {:.7f}\n".format(name, value['threshold'], value['min'], value['max']))
                else:
                    tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join([str(i) for i in value['step']]), 
                            len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                    if name.endswith('_weight') or name.endswith('_bias'):
                        weight_scale.append(tmpstr)
                    else:
                        f.write(tmpstr)
            f.write('#int4_th\n')
            for i in int4_th:
                f.write(i)
            f.write('#weight_scale\n')
            for i in weight_scale:
                f.write(i)
        export_qtable(context_filename, model_name, output_path, cali_mode)

@register_deploy_function(BackendType.Sophgo_TPU)
def deploy_qparams_sophgo_tpu(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for sophgo_tpu.")
    remove_fakequantize_and_collect_params_sophgo(onnx_model_path, model_name)
    output_path = osp.dirname(onnx_model_path)
    context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
    cali_mode = "sophgo_tpu"
    file_h = open(context_filename, "r")
    blob_range = json.loads(file_h.read())[cali_mode]
    file_h.close()
    cali_table = osp.join(output_path, '{}_cali_table_from_mqbench_{}'.format(model_name, cali_mode))
    work_mode = kwargs.get('work_mode', 'QAT_all_int8')
    if work_mode not in  ['QAT_all_int8', 'int4_and_int8_mix', 'int4_and_int8_mix_no_fc']:
        print('QAT_all_int8 not in [QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc],set to QAT_all_int8')
        work_mode = 'QAT_all_int8'
    with open(cali_table, 'w') as f:
        f.write(f"# work_mode:{work_mode} #Automatically generated, do not modify, work_mode choice:[QAT_all_int8, int4_and_int8_mix, int4_and_int8_mix_no_fc]\n")
        f.write("# op_name    threshold    min    max\n")
        weight_scale = []
        int4_th = []
        for name,value in blob_range.items():
            if 'threshold' in value:
                tmpstr = "{} {:.7f} {:.7f} {:.7f}\n".format(name[:-2], value['threshold'], value['min'], value['max'])
                if name.endswith('_4'):
                    int4_th.append(tmpstr)
                elif name.endswith('_8'):
                    f.write(tmpstr)
                else:
                    f.write("{} {:.7f} {:.7f} {:.7f}\n".format(name, value['threshold'], value['min'], value['max']))
            else:
                tmpstr = "{} {} {} {} {}\n".format(name, len(value['step']), ' '.join([str(i) for i in value['step']]), 
                        len(value['zero_point']), ' '.join([str(i) for i in value['zero_point']]))
                if name.endswith('_weight') or name.endswith('_bias'):
                    weight_scale.append(tmpstr)
                else:
                    f.write(tmpstr)
        f.write('#int4_th\n')
        for i in int4_th:
            f.write(i)
        f.write('#weight_scale\n')
        for i in weight_scale:
            f.write(i)
    export_qtable(context_filename, model_name, output_path, cali_mode)


@register_deploy_function(BackendType.Vitis)
def deploy_qparams_vitis(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Vitis-DPU.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='vitis')


@register_deploy_function(BackendType.SNPE)
def deploy_qparams_snpe(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for SNPE.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='snpe')


@register_deploy_function(BackendType.PPLW8A16)
def deploy_qparams_pplw8a16(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for PPLW8A16.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='ppl')


@register_deploy_function(BackendType.ONNX_QNN)
def deploy_qparams_tvm(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Convert to ONNX QNN.")
    ONNXQNNPass(onnx_model_path).run(model_name)


@register_deploy_function(BackendType.PPLCUDA)
def deploy_qparams_ppl_cuda(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for PPL-CUDA.")
    remove_fakequantize_and_collect_params(onnx_model_path, model_name, backend='ppl-cuda')

def convert_deploy(model: GraphModule, backend_type: BackendType,
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
        'deploy_to_qlinear': deploy_to_qlinear
    }
    kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    for convert_function in BACKEND_DEPLOY_FUNCTION[backend_type]:
        convert_function(deploy_model, **kwargs)

def export_onnx_with_fakequant_node(model: GraphModule, backend_type: BackendType,
                   input_shape_dict=None, dummy_input=None, output_path='./',
                   model_name='mqbench_qmodel', deploy_to_qlinear=False, **extra_kwargs):
    r"""Convert GraphModule with fakequant node to onnx

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
        'deploy_to_qlinear': deploy_to_qlinear
    }
    kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    convert_merge_bn(deploy_model, **kwargs)
    convert_onnx(deploy_model, **kwargs)