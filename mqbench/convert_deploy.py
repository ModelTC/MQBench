import json
import os.path as osp
import os
import numpy as np
import torch
from torch.fx import GraphModule
import onnx
from onnxsim import simplify
import mqbench.custom_symbolic_opset  # noqa: F401
import mqbench.fusion_method          # noqa: F401
from mqbench.utils import deepcopy_graphmodule
from .OnnxOpt import onnx_opt
from mqbench.utils.logger import logger
from mqbench.utils.registry import (
    NET_DEPLOY_FUNCTION,
    FUSED_MODULE_CONVERT_FUNCTION,
    register_deploy_function
)
from mqbench.deploy import (
    remove_fakequantize_and_collect_params_nnie,
    remove_fakequantize_and_collect_params,
    remove_fakequantize_and_collect_params_tf,
    remove_fakequantize_and_collect_params_flt,
    replace_fakequantize_and_collect_params_openvino,
    remove_fakequantize_and_collect_params_sophgo,
    ONNXQLinearPass, ONNXQNNPass
)

from mqbench.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
    E4M3FakeQuantize,
    E5M2FakeQuantize,
    GPTQFakeQuantize,
    FP4FakeQuantize,
    GPTQFP4FakeQuantize,
    FP4GROUPFakeQuantize,
    FP4GROUPFakeQuantize1
)
FP8_FAKEQUANTIZER = [E4M3FakeQuantize, E5M2FakeQuantize]
FP4_FAKEQUANTIZER = [FP4FakeQuantize, GPTQFP4FakeQuantize, FP4GROUPFakeQuantize, FP4GROUPFakeQuantize1]
INT_FAKEQUANTIZER = [LearnableFakeQuantize, NNIEFakeQuantize, FixedFakeQuantize, DoReFaFakeQuantize, DSQFakeQuantize, PACTFakeQuantize,\
                     TqtFakeQuantize, AdaRoundFakeQuantize, QDropFakeQuantize, GPTQFakeQuantize]

__all__ = ['convert_deploy']

@register_deploy_function("CNN")
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


@register_deploy_function("CNN")
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
            class_of_submodule = submodule.__class__
            if class_of_submodule in FP8_FAKEQUANTIZER:
                quant_mode = "FP8"

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

     
@register_deploy_function("Transformer")
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
    # opset_version = 11 if kwargs.get('deploy_to_qlinear', False) else 13
    opset_version = 18

    quant_mode = 'INT8'
    # open all fake quant node to export
    if isinstance(model, torch.fx.graph_module.GraphModule):
        # print(">>>>> print graphmodule before export", model)
        # print(">>>>> print graph before export")
        # model.graph.print_tabular()
        for name, submodule in model.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                submodule.disable_observer()
                submodule.enable_fake_quant()
            class_of_submodule = submodule.__class__
            if class_of_submodule in FP8_FAKEQUANTIZER:
                quant_mode = "FP8"
    # skip FP8 fakequant onnx because of the export problem
    if quant_mode != "FP8":
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
    # if it is F8 fakequant, load the existed onnx with INT8 fakequant
    model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)
    model_onnx = onnx.shape_inference.infer_shapes(model_onnx)
    os.system(f"rm -f {onnx_model_path}")
    onnx.save(model_onnx, onnx_model_path)


def export_qtable_tf(context_filename, model_name, output_path, quant_mode):
        print("导出qtable")
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[quant_mode]
        file_h.close()
        q_table = osp.join(output_path, '{}_q_table_from_mqbench_{}'.format(model_name, quant_mode))
        with open(q_table, 'w') as f:
            f.write("# qtable from MQBench\n")
            f.write("# op_name  quantize_mode\n") # match the colnames of qtable in tpu-mlir
            for name,value in blob_range.items():
                if 'quant_type' in value:
                    quant_type = value['quant_type']
                    if 'threshold' in value:
                        f.write("{} {}\n".format(name, dtype))
                        # f.write("{} {}\n".format(name[:-2], quant_type))
                    else:
                        f.write("{} {}\n".format(name, quant_type))
                    continue
                if value['only_observer']==0 or value['only_observer']==1:
                    dtype = 'F32'
                    if 'threshold' in value:
                    # f.write("{} {}\n".format(name[:-2], dtype))
                        f.write("{} {}\n".format(name, dtype))
                    else:
                        f.write("{} {}\n".format(name, dtype))
                    continue
                dtype ='INT8'
                if value['bit'] == 4:
                    dtype = 'INT4'
                elif value['type'] == 'uint':
                    dtype = 'UINT8'
                if 'threshold' in value:
                    # f.write("{} {}\n".format(name[:-2], dtype))
                    f.write("{} {}\n".format(name, dtype))
                else:
                    f.write("{} {}\n".format(name, dtype))

def export_qtable(context_filename, model_name, output_path, quant_mode):
        print("导出qtable")
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[quant_mode]
        file_h.close()
        q_table = osp.join(output_path, '{}_q_table_from_mqbench_{}'.format(model_name, quant_mode))
        with open(q_table, 'w') as f:
            f.write("# qtable from MQBench\n")
            f.write("# op_name  quantize_mode\n") # match the colnames of qtable in tpu-mlir
            for name,value in blob_range.items():
                if 'quant_type' in value:
                    quant_type = value['quant_type']
                    if 'threshold' in value:
                        f.write("{} {}\n".format(name[:-2], quant_type))
                    else:
                        f.write("{} {}\n".format(name, quant_type))
                    continue
                dtype = 'INT8'
                if value['bit'] == 4:
                    dtype = 'INT4'
                elif value['type'] == 'uint':
                    dtype = 'UINT8'
                if 'threshold' in value:
                    f.write("{} {}\n".format(name[:-2], dtype))
                else:
                    f.write("{} {}\n".format(name, dtype))
                    
                    
@register_deploy_function("Transformer")
def deploy_qparams_Academic_NLP(model: GraphModule, onnx_model_path, model_name, **kwargs):
    logger.info("Extract qparams for Academic_NLP.")
    node_name_only_observer=[]
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if submodule.only_enable_observer==True:
                node_name_only_observer.append(name)  
    quant_mode = "INT8"
    for name, submodule in model.named_modules():
        class_of_submodule = submodule.__class__
        if class_of_submodule in FP8_FAKEQUANTIZER:
            quant_mode = "FP8"
    cali_mode = "Academic_NLP"
    if quant_mode == "FP8":
        remove_fakequantize_and_collect_params_flt(onnx_model_path, model_name, backend='Academic_NLP')
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[cali_mode+"_Float"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_float_cali_table_from_mqbench_Academic_NLP'.format(model_name))
        fp8_header = "mqbench-fp8"
        with open(cali_table, 'w') as f:
            f.write(f"# work_mode:{fp8_header} #Automatically generated, do not modify, work_mode choice:[E4M3_RNE, E5M2_RNE]\n")
            f.write("#       op_name        threshold        min        max\n")
            weight_scale_fp8 = []
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
            f.write('#weight_scale_fp8\n')
            for i in weight_scale_fp8:
                f.write(i)
        cali_mode_new = cali_mode + '_Float'
        export_qtable_tf(context_filename, model_name, output_path, cali_mode_new)
    else:
        remove_fakequantize_and_collect_params_tf(onnx_model_path, model_name, backend='Academic_NLP',node_name_only_observer=node_name_only_observer)
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
        export_qtable_tf(context_filename, model_name, output_path, cali_mode)

@register_deploy_function("CNN")
def deploy_qparams_sophgo_tpu(model: GraphModule, onnx_model_path, model_name, quant_type_dict, **kwargs):
    logger.info("Extract qparams for sophgo_tpu.")
    quant_values_list = list(quant_type_dict.values())
    quant_mode = "INT8"
    if 'FP8' in quant_values_list:
        quant_mode = "FP8"
    cali_mode = "sophgo_tpu"
    if quant_mode == "FP8":
        remove_fakequantize_and_collect_params_flt(onnx_model_path, model_name, backend='sophgo_tpu')
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())[cali_mode+"_Float"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_float_cali_table_from_mqbench_sophgo_tpu'.format(model_name))
        fp8_header = "mqbench-fp8"
        with open(cali_table, 'w') as f:
            f.write(f"# work_mode:{fp8_header} #Automatically generated, do not modify, work_mode choice:[E4ME_RNE, E5M2_RNE]\n")
            f.write("#       op_name        threshold        min        max\n")
            weight_scale_fp8 = []
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
            f.write('#weight_scale_fp8\n')
            for i in weight_scale_fp8:
                f.write(i)
        cali_mode_new = cali_mode + "_Float"
        export_qtable(context_filename, model_name, output_path, cali_mode_new)
    else:
        remove_fakequantize_and_collect_params_sophgo(onnx_model_path, model_name, quant_type_dict)
        print("导出calitable")
        output_path = osp.dirname(onnx_model_path)
        context_filename = osp.join(output_path, '{}_clip_ranges.json'.format(model_name))
        file_h = open(context_filename, "r")
        blob_range = json.loads(file_h.read())["sophgo_tpu"]
        file_h.close()
        cali_table = osp.join(output_path, '{}_cali_table_from_mqbench_sophgo_tpu'.format(model_name))
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

def get_quant_type_from_fakequant_type(model: GraphModule):
    r"""
    Given GraphModule, Traverse each fakequant node within it,
    obtain the quantization type based on the class of the fakequant node.
    """
    quant_type_dict = {}
    for name, submodule in model.named_modules(remove_duplicate=False):
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if submodule.only_enable_observer:
                quant_type_dict[name] = 'F32'
            else:
                class_of_submodule = submodule.__class__
                if class_of_submodule in INT_FAKEQUANTIZER:
                    qmax = submodule.quant_max
                    qmin = submodule.quant_min
                    bit = int(np.log2(qmax-qmin+1))
                    if (bit == 8 and qmin < 0):
                        quant_type_dict[name] = 'INT8'
                    elif (bit == 8 and qmin ==0):
                        quant_type_dict[name] = 'UINT8'
                    elif (bit == 4 and qmin < 0):
                        quant_type_dict[name] = 'INT4'
                elif class_of_submodule in FP4_FAKEQUANTIZER:
                    quant_type_dict[name] = 'FP4'
                elif class_of_submodule in FP8_FAKEQUANTIZER:
                    quant_type_dict[name] = 'FP8'
                else:
                    quant_type_dict[name] = 'None'

    return quant_type_dict

def convert_deploy(model: GraphModule, net_type='CNN',
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
    quant_type_dict = get_quant_type_from_fakequant_type(model)
    kwargs = {
        'input_shape_dict': input_shape_dict,
        'dummy_input': dummy_input,
        'output_path': output_path,
        'model_name': model_name,
        'onnx_model_path': osp.join(output_path, '{}.onnx'.format(model_name)),
        'deploy_to_qlinear': deploy_to_qlinear,
        'quant_type_dict': quant_type_dict
    }
    kwargs.update(extra_kwargs)
    deploy_model = deepcopy_graphmodule(model)
    for convert_function in NET_DEPLOY_FUNCTION[net_type]:
        convert_function(deploy_model, **kwargs)

def export_onnx_with_fakequant_node(model: GraphModule, net_type='CNN',
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