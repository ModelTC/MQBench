import json
import os
import onnx
from onnx import numpy_helper
import numpy as np
from mqbench.utils.logger import logger

perchannel_fakequantizer = ['FakeQuantizeLearnablePerchannelAffine', 'FixedPerChannelAffine', 'FakeQuantizeDSQPerchannel']
pertensor_fakequantizer = ['LearnablePerTensorAffine', 'FixedPerTensorAffine', 'FakeQuantizeDSQPertensor']
all_fakequantizer = perchannel_fakequantizer + pertensor_fakequantizer

def update_inp2node_out2node(graph):
    out2node = {}
    inp2node = {}
    for node in graph.node:
        for out in node.output:
            # suppose each node only has one output
            out2node[out] = node
        for idx, inp in enumerate(node.input):
            # one node may have multiple inputs
            if inp not in inp2node:
                inp2node[inp] = []
            inp2node[inp].append([node, idx])
    return out2node, inp2node

def prepare_data(graph):
    params = {}
    for init in graph.initializer:
        params[init.name] = numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    params[node.output[0]] = numpy_helper.to_array(attr.t)
    return params

def prepare_initializer(graph):
    named_initializer = {}
    for init in graph.initializer:
        named_initializer[init.name] = init
    return named_initializer

def parse_attrs(node_attrs):
    attrs = {}
    for attr in node_attrs:
        if attr.type == onnx.AttributeProto.AttributeType.INTS:
            attrs[attr.name] = tuple(attr.ints)
        elif attr.type == onnx.AttributeProto.AttributeType.INT:
            attrs[attr.name] = attr.i
        elif attr.type == onnx.AttributeProto.AttributeType.FLOATS:
            attrs[attr.name] = tuple(attr.floats)
        elif attr.type == onnx.AttributeProto.AttributeType.FLOAT:
            attrs[attr.name] = attr.f
        elif attr.type == onnx.AttributeProto.AttributeType.TENSOR:
            attrs[attr.name] = numpy_helper.to_array(attr.t)
        elif attr.type == onnx.AttributeProto.AttributeType.STRING:
            attrs[attr.name] = str(attr.s)
        elif attr.type == onnx.AttributeProto.AttributeType.STRINGS:
            attrs[attr.name] = tuple([str(x) for x in attr.strings])
        else:
            raise Exception("ATTR Type [{}] Not Supported!".format(attr.type))
    return attrs

def get_constant_inputs(node, out2node):
    node_list = []
    for inp in node.input:
        if inp in out2node and out2node[inp].op_type == 'Constant':
            node_list.append(out2node[inp])
    return node_list

class OnnxPreprocess(object):
    def replace_resize_op_with_upsample(self, graph, out2node):
        nodes_to_be_removed = []
        idx = 0
        while idx < len(graph.node):
            node = graph.node[idx]
            if node.op_type == 'Resize':
                logger.info(f"Replace resize op: <{node.name}> with upsample.")
                mode = 'nearest'
                for attr in node.attribute:
                    if attr.name == 'mode':
                        mode = attr.s
                upsample_node = onnx.helper.make_node('Upsample',
                                                      name=node.name,
                                                      inputs=[node.input[0], node.input[2]],
                                                      outputs=node.output,
                                                      mode=mode)
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))
                graph.node.insert(idx, upsample_node)
                idx += 1
            idx += 1
        for node in nodes_to_be_removed:
            graph.node.remove(node)
        return

    def remove_fake_pad_op(self, graph, name2data, inp2node, out2node):
        nodes_to_be_removed = []
        for idx, node in enumerate(graph.node):
            node = graph.node[idx]
            if node.op_type == 'Pad':
                pads = name2data[node.input[1]]
                if all([x == 0 for x in pads]):
                    logger.info(f"Remove pad op: <{node.name}>.")
                    next_nodes = inp2node[node.output[0]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]
                    nodes_to_be_removed.append(node)
                    nodes_to_be_removed.extend(get_constant_inputs(node, out2node))
        for node in nodes_to_be_removed:
            graph.node.remove(node)
        return

class NNIE_process(object):
    def gen_gfpq_param_file(self, graph, clip_val):
        nnie_exclude_layer_type = ['Flatten', 'Relu', 'PRelu', 'Sigmoid', 'Reshape',
                                   'Softmax', 'CaffeSoftmax', 'Clip', 'GlobalAveragePool', 'Mul']
        interp_layer_cnt = 0
        gfpq_param_dict = {}
        for idx, node in enumerate(graph.node):
            # We can not support NNIE group conv.
            # Group conv need group-size input params.
            if node.op_type == 'Conv' and node.attribute[1].i != 1:
                continue

            layer_input_tensor = []
            for in_tensor in node.input:
                if in_tensor in clip_val:
                    clip_value = clip_val[in_tensor]
                    layer_input_tensor.append(float(clip_value))
                # Upsample layer only reserve one input.
                if node.op_type in ['Upsample', 'DynamicUpsample']:
                    break

            if node.op_type not in nnie_exclude_layer_type and len(layer_input_tensor) > 0:
                gfpq_param_dict[node.name] = layer_input_tensor

            # Upsample ---> Upsample + Permute in NNIE.
            if node.op_type in ['Upsample', 'DynamicUpsample']:
                interp_layer_name = node.name
                gfpq_param_dict[interp_layer_name + '_permute_' + str(interp_layer_cnt)] = gfpq_param_dict[interp_layer_name]
                interp_layer_cnt += 1
        return gfpq_param_dict

    def remove_fakequantize_and_collect_params(self, onnx_path):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.replace_resize_op_with_upsample(graph, out2node)
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        nodes_to_be_removed = []
        clip_ranges = {}
        for node in graph.node:
            if node.op_type == 'NNIEQuantize':
                next_nodes = inp2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    next_node, idx = next_nodes[0]
                    next_node.input[idx] = node.input[0]
                    # clip weights
                    tensor_name = node.input[0]
                    data = name2data[tensor_name]
                    clip_range = name2data[node.input[1]]
                    new_data = np.clip(data, -clip_range, clip_range)
                    new_data = numpy_helper.from_array(new_data)
                    named_initializer[tensor_name].raw_data = new_data.raw_data
                    logger.info(f'Clip weights {tensor_name} to range [{-clip_range}, {clip_range}].')
                else:
                    # fake quantize for activations
                    clip_ranges[node.input[0]] = name2data[node.input[1]]
                    for next_node, idx in next_nodes:
                        next_node.input[idx] = node.input[0]

                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))

        for node in nodes_to_be_removed:
            graph.node.remove(node)

        gfpq_param_dict = self.gen_gfpq_param_file(graph, clip_ranges)

        output_path = os.path.dirname(onnx_path)
        filename = os.path.join(output_path, 'nnie_gfpq_param_dict.json')
        with open(filename, 'w') as f:
            json.dump({"nnie": {"gfpq_param_dict": gfpq_param_dict}}, f, indent=4)
        filename = os.path.join(output_path, 'nnie_deploy_model.onnx')
        onnx.save(model, filename)
        logger.info("Finish deploy process.")

remove_fakequantize_and_collect_params_nnie = NNIE_process().remove_fakequantize_and_collect_params

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
        assert next_node.op_type in ['Conv', 'Gemm']
        redundant_nodes = []
        if node.input[0] not in named_initializer:
            node.input[0], redundant_nodes = \
                self.weight_preprocess(node.input[0], out2node, inp2node, named_initializer)
        next_node.input[idx] = node.input[0]
        return redundant_nodes

    def deal_with_activation_fakequant(self, node, inp2node):
        next_nodes = inp2node[node.output[0]]
        for next_node, idx in next_nodes:
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

    def clip_weight(self, node, name2data, named_initializer):
        tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
        data = name2data[tensor_name]
        clip_range_min = (qmin - zero_point) * scale
        clip_range_max = (qmax - zero_point) * scale
        if scale.shape[0] > 1:
            new_data = []
            for c in range(data.shape[0]):
                new_data.append(np.clip(data[c], clip_range_min[c], clip_range_max[c]))
            new_data = np.array(new_data)
            logger.info(f'Clip weights <{tensor_name}> to per-channel ranges.')
        else:
            new_data = np.clip(data, clip_range_min, clip_range_max)
            logger.info(f'Clip weights <{tensor_name}> to range [{clip_range_min}, {clip_range_max}].')
        new_data = numpy_helper.from_array(new_data)
        named_initializer[tensor_name].raw_data = new_data.raw_data

    def post_process_clip_ranges(self, clip_ranges, graph, inp2node):
        def find_the_closest_clip_range(node):
            if node.input[0] in clip_ranges:
                return node.input[0]
            elif node.op_type in ['Flatten', 'Resize'] and node.output[0] in inp2node:
                return find_the_closest_clip_range(inp2node[node.output[0]][0][0])
            else:
                return None

        for node in graph.node:
            if node.op_type in ['Flatten', 'Resize']:
                tensor_name = find_the_closest_clip_range(node)
                if tensor_name:
                    clip_ranges[node.input[0]] = clip_ranges[tensor_name]
                    logger.info(f'Pass <{tensor_name}> clip range to <{node.name}> input <{node.input[0]}>.')
        return clip_ranges

    def remove_fakequantize_and_collect_params(self, onnx_path, backend):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        preprocess = OnnxPreprocess()
        preprocess.replace_resize_op_with_upsample(graph, out2node)
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        clip_ranges = {}
        nodes_to_be_removed = []
        for node in graph.node:
            if node.op_type in all_fakequantizer:
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))

            if node.op_type in perchannel_fakequantizer:
                # fake quantize for weights, suppose per-channel quantize only for weight
                redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                nodes_to_be_removed.extend(redundant_nodes)
                self.clip_weight(node, name2data, named_initializer)
                if backend == 'ppl':
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    clip_ranges[tensor_name] = {'step': [float(x) for x in scale],
                                                'zero_point': [int(x) for x in zero_point],
                                                'min': [float(x) for x in scale * (qmin - zero_point)],
                                                'max': [float(x) for x in scale * (qmax - zero_point)],
                                                'bit': int(np.log2(qmax - qmin + 1)),
                                                'type': "biased",
                                                }

            elif node.op_type in pertensor_fakequantizer:
                if node.output[0] in [x.name for x in graph.output]:
                    inp2node[node.output[0]] = []

                next_nodes = inp2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, name2data, named_initializer)
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, inp2node)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    for out in graph.output:
                        if out.name == node.output[0]:
                            out.name = tensor_name

                    if backend == 'tensorrt':
                        clip_ranges[tensor_name] = float(scale * min(-qmin, qmax))
                    elif backend == 'snpe':
                        clip_ranges[tensor_name] = [
                            {'bitwidth': int(np.log2(qmax - qmin + 1)),
                             'min': float(scale * (qmin - zero_point)),
                             'max': float(scale * (qmax - zero_point))}
                        ]
                if backend == 'ppl':
                    clip_ranges[tensor_name] = {'step': float(scale),
                                                'zero_point': int(zero_point),
                                                'min': float(scale * (qmin - zero_point)),
                                                'max': float(scale * (qmax - zero_point)),
                                                'bit': int(np.log2(qmax - qmin + 1)),
                                                'type': "biased",
                                                }

        for node in nodes_to_be_removed:
            graph.node.remove(node)

        clip_ranges = self.post_process_clip_ranges(clip_ranges, graph, inp2node)
        if backend == 'tensorrt':
            context = {"tensorrt": {"blob_range": clip_ranges}}
        elif backend == 'snpe':
            context = {'activation_encodings': clip_ranges, 'param_encodings': {}}
        elif backend == 'ppl':
            context = {"ppl": clip_ranges}
        output_path = os.path.dirname(onnx_path)
        filename = os.path.join(output_path, '{}_clip_ranges.json'.format(backend))
        with open(filename, 'w') as f:
            json.dump(context, f, indent=4)
        filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(backend))
        onnx.save(model, filename)
        logger.info("Finish deploy process.")

remove_fakequantize_and_collect_params = LinearQuantizer_process().remove_fakequantize_and_collect_params
