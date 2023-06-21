import json
import os
import copy


import onnx
import numpy as np
from onnx import numpy_helper

from mqbench.utils.logger import logger
from mqbench.deploy.common import (
    update_inp2node_out2node,
    prepare_initializer,
    prepare_data,
    OnnxPreprocess,
    get_constant_inputs,
    parse_attrs
)


PERCHANNEL_FAKEQUANTIZER = ['FakeQuantizeLearnablePerchannelAffine', 
                            'FixedPerChannelAffine',
                            'FakeQuantizeDSQPerchannel']
PERTENSOR_FAKEQUANTIZER = ['LearnablePerTensorAffine', 
                           'FixedPerTensorAffine',
                           'FakeQuantizeDSQPertensor',
                           'FakeQuantizeTqtAffine']
ALL_FAKEQUANTIZER = PERCHANNEL_FAKEQUANTIZER + PERTENSOR_FAKEQUANTIZER


class LinearQuantizer_process(object):
    # some method like dorefa need pre-compute weights
    def weight_preprocess(self, target_tensor, out2node, inp2node, named_initializer):
        def find_weight(tensor):
            if tensor not in named_initializer:
                _node = out2node[tensor]
                for inp in _node.input:
                    return find_weight(inp)
            return tensor
        weight = find_weight(target_tensor)

        # TODO need more general method, like onnxruntime infer
        data = numpy_helper.to_array(named_initializer[weight])
        data = np.tanh(data)
        data = data / (np.max(np.abs(data)) + 1e-5)
        data = numpy_helper.from_array(data)
        named_initializer[weight].raw_data = data.raw_data

        redundant_nodes = []

        def find_redundant_nodes(tensor):
            if tensor == target_tensor:
                return
            nodes = inp2node[tensor]
            for node, idx in nodes:
                if node not in redundant_nodes:
                    redundant_nodes.append(node)
                    redundant_nodes.extend(get_constant_inputs(node, out2node))
                find_redundant_nodes(node.output[0])
        find_redundant_nodes(weight)
        return weight, redundant_nodes

    def deal_with_weight_fakequant(self, node, out2node, inp2node, named_initializer):
        next_nodes = inp2node[node.output[0]]
        assert len(next_nodes) == 1
        next_node, idx = next_nodes[0]
        assert next_node.op_type in ['Conv', 'Gemm', 'ConvTranspose']
        redundant_nodes = []
        if node.input[0] not in named_initializer:
            node.input[0], redundant_nodes = \
                self.weight_preprocess(node.input[0], out2node, inp2node, named_initializer)
        next_node.input[idx] = node.input[0]
        return redundant_nodes

    def deal_with_activation_fakequant(self, node, inp2node):
        next_nodes = inp2node[node.output[0]]
        for next_node, idx in next_nodes:
            # if next_node.op_type == 'Add' and next_node.output[0] in inp2node: #将observer qdq的输入写到下一个layer的输入
            #     nextnode, i = inp2node[next_node.output[0]][0]
            #     nextnode.input[i] = node.input[0]
            # else:
            next_node.input[idx] = node.input[0]
        return

    def parse_qparams(self, node, name2data):
        tensor_name, scale, zero_point = node.input[:3]
        scale, zero_point = name2data[scale], name2data[zero_point]
        if len(node.input) > 3:
            qmin, qmax = node.input[-2:]
            qmin, qmax = name2data[qmin], name2data[qmax]
        elif len(node.attribute) > 0:
            qparams = parse_attrs(node.attribute)
            qmin = qparams['quant_min']
            qmax = qparams['quant_max']
        else:
            logger.info(f'qmin and qmax are not found for <{node.name}>!')
        return tensor_name, scale, zero_point, qmin, qmax

    def clip_weight(self, node, name2data, inp2node, named_initializer):
        tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
        data = name2data[tensor_name]
        clip_range_min = ((qmin - zero_point) * scale).astype(data.dtype)
        clip_range_max = ((qmax - zero_point) * scale).astype(data.dtype)
        if len(scale.shape) > 0 and scale.shape[0] > 1:
            new_data = []
            transposed = False
            next_node = inp2node[node.output[0]]
            if len(next_node) == 1 and next_node[0][0].op_type == 'ConvTranspose':
                transposed = True
                data = data.transpose(1, 0, 2, 3)
            for c in range(data.shape[0]):
                new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            if transposed:
                new_data = new_data.transpose(1, 0, 2, 3)
            logger.info(f'Clip weights <{tensor_name}> to per-channel ranges.')
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
            logger.info(f'Clip weights <{tensor_name}> to range [{clip_range_min}, {clip_range_max}].')
        new_data = numpy_helper.from_array(new_data)
        named_initializer[tensor_name].raw_data = new_data.raw_data
    
    def get_correct_sophgo_tpu_input_tensor_name(self, node, out2node, have_int4=False): #和tpu-mlir的命名风格一致
        input_0 = node.input[0]
        op_type = ''
        if input_0 in out2node:
            op_type = out2node[input_0].op_type
            tensor_name = '{}_{}'.format(input_0, op_type)
        else:
            tensor_name = input_0
        if have_int4 and op_type in ['Conv', "Gemm"]:
            tensor_name += '_4'
        else:
            tensor_name += '_8'
        return tensor_name

    def post_process_clip_ranges(self, clip_ranges, graph, inp2node, out2node):
        def find_the_closest_clip_range(node):
            tensor_name = self.get_correct_sophgo_tpu_input_tensor_name(node, out2node)
            if tensor_name in clip_ranges:
                return tensor_name
            elif node.op_type in ['Flatten', 'Resize'] and node.output[0] in inp2node:
                return find_the_closest_clip_range(inp2node[node.output[0]][0][0])
            else:
                return None

        for node in graph.node:
            if node.op_type in ['Flatten', 'Resize']:
                tensor_name = find_the_closest_clip_range(node)
                if tensor_name:
                    new_name = self.get_correct_sophgo_tpu_input_tensor_name(node, out2node)
                    clip_ranges[new_name] = copy.deepcopy(clip_ranges[tensor_name])
                    logger.info(f'Pass <{tensor_name}> clip range to <{node.name}> input <{node.input[0]}>.')
        return clip_ranges

    def post_process_clip_ranges2(self, clip_ranges, graph, inp2node, out2node, have_int4 = False):
        op_type_inAndOutShouldSameClipRange = ['Flatten', 'Resize', 'Reshape', 'Transpose']
        for node in graph.node:
            tensor_name = f'{node.output[0]}_{node.op_type}'
            tensor_name += '_4' if have_int4 and node.op_type in ['Conv', "Gemm"] else '_8'
            if tensor_name not in clip_ranges:
                pre_op = node
                finded = False
                while pre_op.op_type in op_type_inAndOutShouldSameClipRange:
                    tensor_name2 = self.get_correct_sophgo_tpu_input_tensor_name(pre_op, out2node, have_int4)
                    if tensor_name2 in clip_ranges:
                        finded = True
                        clip_ranges[tensor_name] = clip_ranges[tensor_name2]
                        print(f'pre_op finded, transfer {tensor_name2} to {tensor_name}')
                        break
                    if pre_op.input[0] in out2node:
                        pre_op = out2node[pre_op.input[0]]
                    else:
                        print(f'{pre_op.name}\'s pre_node not exist')
                        break
                if not finded:
                    if node.output[0] in inp2node:
                        next_op = inp2node[node.output[0]][0][0]
                        while next_op.op_type in op_type_inAndOutShouldSameClipRange:
                            tensor_name2 = f'{next_op.output[0]}_{next_op.op_type}'
                            tensor_name2 += '_4' if have_int4 and next_op.op_type in ['Conv', "Gemm"] else '_8'
                            if tensor_name2 in clip_ranges:
                                finded = True
                                clip_ranges[tensor_name] = clip_ranges[tensor_name2]
                                print(f'next_op finded, transfer {tensor_name2} to {tensor_name}')
                                break
                            if next_op.output[0] in inp2node:
                                next_op = inp2node[next_op.output[0]][0][0]
                            else:
                                print(f'{next_op.name}\'s next_op not exist')
                                break
                    else:
                        print(f'{node.name}\'s next_op not exist')
                # if not finded:
                #     print(f'Waring:{node.name}\'s clip_ranges not exist, maybe have some error')
        return clip_ranges

    def isQdqAdd(self, tensor_name, out2node):
        pre_name = None
        for i in out2node[tensor_name].input:
            if out2node[i].op_type == 'LearnablePerTensorAffine':
                pre_name = out2node[i].input[0]
                if pre_name in out2node:
                    pre_name = '{}_{}'.format(pre_name, out2node[pre_name].op_type)
            if  out2node[i].op_type not in ['LearnablePerTensorAffine', 'Sub']:
                return False, pre_name
        return True, pre_name

    def remove_Qdq_add_sub(self, graph, inp2node, out2node):
        nodes_to_be_removed = []
        for idx, node in enumerate(graph.node):
            if node.op_type == 'Add':
                isQdqAdd, _ = self.isQdqAdd(node.output[0], out2node)
                if isQdqAdd:
                    for i in node.input:
                        if i.op_type == 'Sub':
                            nodes_to_be_removed.append(out2node[i]) 
                        else:
                            fake_quant_output = i
                    next_nodes = inp2node[node.output[0]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = fake_quant_output                                 
                    nodes_to_be_removed.append(node)         
                    nodes_to_be_removed.extend(get_constant_inputs(node, out2node))
        for node in nodes_to_be_removed:
            graph.node.remove(node)
        return

    def remove_fakequantize_and_collect_params(self, onnx_path, model_name):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)
        # self.remove_Qdq_add_sub(graph, inp2node, out2node)
        # out2node, inp2node = update_inp2node_out2node(graph)

        clip_ranges = {}
        nodes_to_be_removed = []
        output_path = os.path.dirname(onnx_path)
        file_name = os.path.join(output_path, 'layer_outputs.npz')
        layer_out_tensor = None
        have_int4 = False
        layer_out_tensor2 = {}
        if os.path.exists(file_name):
            layer_out_tensor = np.load(file_name)
        for node in graph.node:
            print(f'process node:{node.name}, type:{node.op_type}')
            if node.op_type in ALL_FAKEQUANTIZER:
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))
            if node.output[0] not in inp2node:
                assert node.output[0] in [l.name for l in graph.output]
                inp2node[node.output[0]] = []
            next_nodes = inp2node[node.output[0]]
            if node.op_type in PERCHANNEL_FAKEQUANTIZER:
                # fake quantize for weights, suppose per-channel quantize only for weight
                redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node, name2data, inp2node, named_initializer)
                tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                #卷积权重per-channel量化参数，bias的per-chan量化参数没有去调优
                if len(next_nodes) == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:#当前伪量化节点只有1个后继，且第1个后继节点为conv类型
                    next_node_output = next_nodes[0][0].output[0]
                    if inp2node[next_node_output][0][0].op_type == 'Relu':##伪量化节点的第1个后继conv节点的第1个后继节点为Relu(fake->conv->relu)
                        #若是fake->conv->relu,因为relu会融合到前面conv，故用relu的输出tensor名+Relu作为量化参数保存tensor名
                        tensor_name = '{}_{}'.format(inp2node[next_node_output][0][0].output[0], 'Relu')
                    else:
                        #若是fake->conv->not_relu_type,直接用conv的输出tensor名+conv作为量化参数保存tensor名
                        tensor_name = '{}_{}'.format(next_node_output, next_nodes[0][0].op_type)
                    tensor_name += '_{}'.format('weight' if next_nodes[0][1] == 1 else 'bias'  )
                    clip_ranges[tensor_name] = {'step': [float(x) for x in scale],
                                                'zero_point': [int(x) for x in zero_point]}
            elif node.op_type in PERTENSOR_FAKEQUANTIZER:
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, name2data, inp2node, named_initializer)
                    assert next_nodes[0][0].op_type == 'Gemm'
                    tensor_name_new = '{}_{}_weight'.format(next_nodes[0][0].output[0], next_nodes[0][0].op_type)
                    clip_ranges[tensor_name_new] = {'step': [float(x) for x in scale],
                                                'zero_point': [int(x) for x in zero_point]}
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, inp2node)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    bits = 4 if qmax == 7 else 8
                    if bits == 4:
                        have_int4 = True
                    for out in graph.output:
                        output_name = node.output[0]
                        # if inp2node[node.output[0]][0][0].op_type == 'Add':
                        #     output_name = inp2node[node.output[0]][0][0].output[0]
                        if out.name == output_name:
                            out.name = tensor_name

                    scale_name = node.input[1]
                    post_str = '_post_act_fake_quantizer.scale'
                    tensor_name_new = tensor_name
                    if tensor_name in out2node:
                        tensor_name_new += '_{}'.format(out2node[tensor_name].op_type)
                        # if out2node[tensor_name].op_type == 'Add':
                        #     isQdqAdd, pre_name = self.isQdqAdd(tensor_name, out2node)
                        #     if isQdqAdd:
                        #         tensor_name_new = pre_name+'_qdq'

                        # if out2node[tensor_name].op_type in ['Conv', 'Gemm']:
                        #     if bits == 8:
                        #         tensor_name_new += '_8'
                        # else:
                        #     if bits == 4 and out2node[out2node[tensor_name].input[0]].op_type not in ['Conv', 'Gemm']:
                        #         tensor_name_new += '_4'
                    if layer_out_tensor is not None and scale_name.endswith(post_str):
                        torch_name = scale_name[:len(scale_name)-len(post_str)]
                        if torch_name in layer_out_tensor.files:
                            layer_out_tensor2[tensor_name_new] = layer_out_tensor[torch_name]
                    clip_ranges[tensor_name_new+f'_{bits}'] = {'threshold':float(scale * max(-qmin, qmax)), #对称量化时这个参数生效
                                                'min': float(scale * (qmin - zero_point)),
                                                'max': float(scale * (qmax - zero_point))}
        if layer_out_tensor is not None and len(layer_out_tensor2) > 0:
            if 'data' in layer_out_tensor.files:
                layer_out_tensor2['data'] = layer_out_tensor['data']
            os.system(f'rm -f {file_name}')
            np.savez(file_name, **layer_out_tensor2)
        for node in nodes_to_be_removed:
            graph.node.remove(node)
        # delete initializer
        out2node, inp2node = update_inp2node_out2node(graph)
        named_initializer = prepare_initializer(graph)
        for name, initial_data in named_initializer.items():
            if name in (out2node.keys() | inp2node.keys()):
                continue
            graph.initializer.remove(initial_data)
        
        # clip_ranges = self.post_process_clip_ranges(clip_ranges, graph, inp2node, out2node)
        clip_ranges = self.post_process_clip_ranges2(clip_ranges, graph, inp2node, out2node, have_int4)
        context = {'sophgo_tpu': clip_ranges}
        context['w_qscheme'] = ''
        context['a_qscheme'] = ''
        context_filename = os.path.join(output_path, '{}_clip_ranges.json'.format(model_name))
        with open(context_filename, 'w') as f:
            json.dump(context, f, indent=4)
        onnx_filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(model_name))
        model_onnx = onnx.shape_inference.infer_shapes(model)
        os.system(f"rm -f {onnx_filename}")
        onnx.save(model_onnx, onnx_filename)
        logger.info("Finish deploy process.")

remove_fakequantize_and_collect_params_sophgo = LinearQuantizer_process().remove_fakequantize_and_collect_params
