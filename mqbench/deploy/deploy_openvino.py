import os

import onnx
import numpy as np
from onnx import numpy_helper

from onnx import helper

from mqbench.utils.logger import logger
from mqbench.deploy.common import (
    update_inp2node_out2node,
    prepare_data,
    OnnxPreprocess,
    ONNXGraph,
    get_constant_inputs,
    parse_attrs
)

ALL_FAKEQUANTIZER = ['QuantizeLinear', 'DequantizeLinear']

def get_dequant_node(node, inp2node):
    dequant_node = inp2node[node.output[0]][0][0]
    if dequant_node.op_type == 'Clip':
        dequant_node = inp2node[dequant_node.output[0]][0][0]
    assert dequant_node.op_type == 'DequantizeLinear', "This is not correct fakequant node!"
    return dequant_node

class OPENVINO_process(object):

    def parse_qparams(self, node, name2data):
        tensor_name, scale, zero_point = node.input[:3]
        scale, zero_point = name2data[scale], name2data[zero_point]
        # if len(node.input) > 3:
        #     qmin, qmax = node.input[-2:]
        #     qmin, qmax = name2data[qmin], name2data[qmax]
        # elif len(node.attribute) > 0:
        #     qparams = parse_attrs(node.attribute)
        #     qmin = qparams['quant_min']
        #     qmax = qparams['quant_max']
        # else:
        #     logger.info(f'qmin and qmax are not found for <{node.name}>!')
        #     qmax = qmin = None
        return tensor_name, scale, zero_point

    def replace_fakequantize_and_collect_params(self, onnx_path, model_name, qmin_max_dict):
        onnx_graph = ONNXGraph(onnx_path)
        model = onnx_graph.model
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)

        preprocess = OnnxPreprocess()
        preprocess.remove_fake_pad_op(graph, name2data, inp2node, out2node)
        out2node, inp2node = update_inp2node_out2node(graph)

        nodes_to_be_removed = []
        node_defs = []
        insert_initializer_names = set()
        for node in graph.node:
            if node.op_type in ALL_FAKEQUANTIZER:
                next_node = inp2node[node.output[0]][0][0]
                if next_node.op_type == 'Clip' and inp2node[next_node.output[0]][0][0].op_type == 'DequantizeLinear':
                    nodes_to_be_removed.append(next_node)
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))
            if node.op_type == 'QuantizeLinear':
                qmin, qmax = qmin_max_dict[node.name]
                dequant_node = get_dequant_node(node, inp2node)
                tensor_name, scale, zero_point = self.parse_qparams(node, name2data)
                qmax = int(qmax)
                qmin = int(qmin)
                levels = qmax - qmin + 1
                # adjust weight levels
                if levels == 128:
                    levels = 256
                    qmax = qmax * 2 + 1
                    qmin = qmin * 2
                output_name = dequant_node.output[0]
                # Create a node (FakeQuantize)
                fakeq_inputnames = [item % tensor_name for item in ['input_min_%s', 'input_max_%s', 'output_min_%s', 'output_max_%s']]
                node_def = helper.make_node(
                    'FakeQuantize',  # node name
                    [tensor_name, *fakeq_inputnames],  # inputs
                    [output_name],  # outputs
                    levels=levels,  # Attributes
                    domain="org.openvinotoolkit",
                    name=node.name
                )
                node_defs.append(node_def)
                scale = np.abs(np.asarray(scale, dtype=np.float64).reshape(-1))
                zero_point = np.clip(np.asarray(np.round(zero_point), dtype=np.int32).reshape(-1), a_min=qmin, a_max=qmax)

                qrange = float(qmax - qmin)
                input_range = scale * qrange
                input_high = (qmax - zero_point).astype(np.float64) * input_range / qrange
                input_low = input_high - input_range
                input_low_size = input_low.size

                try:
                    next_node = inp2node[dequant_node.output[0]][0][0]
                    # node for save weights
                    fake_node = out2node[next_node.input[1]]
                    tensor = name2data[fake_node.input[0]]
                    shape_length = len(tensor.shape)
                    new_shape = [-1, ] + [1, ] * (shape_length - 1)
                except Exception as e:
                    new_shape = [-1, ]

                if input_low_size != 1:
                    input_low = input_low.reshape(*new_shape)
                    input_high = input_high.reshape(*new_shape)
                input_low = input_low.astype(np.float32)
                input_high = input_high.astype(np.float32)
                for initializer_name, value_tensor in zip(fakeq_inputnames, [input_low, input_high, input_low, input_high]):
                    if initializer_name in insert_initializer_names:
                        continue
                    initializer = numpy_helper.from_array(value_tensor)
                    initializer.name = initializer_name
                    insert_initializer_names.add(initializer_name)
                    graph.initializer.append(initializer)

        for node in nodes_to_be_removed:
            if node in graph.node:
                graph.node.remove(node)
        graph.node.extend(node_defs)
        onnx_graph.topologize_graph()
        onnx_graph.prepare_initializer()
        onnx_graph.optimize_model()
        output_path = os.path.dirname(onnx_path)
        onnx_filename = os.path.join(output_path, '{}_deploy_model.onnx'.format(model_name))
        onnx.save(model, onnx_filename)
        logger.info("Finish deploy process.")


replace_fakequantize_and_collect_params_openvino = OPENVINO_process().replace_fakequantize_and_collect_params