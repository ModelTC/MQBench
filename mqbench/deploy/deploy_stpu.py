import json
import os
from collections import OrderedDict

import onnx

from mqbench.deploy.common import (get_constant_inputs, prepare_data,
                                   prepare_initializer,
                                   update_inp2node_out2node)
from mqbench.deploy.deploy_linear import (PERTENSOR_FAKEQUANTIZER,
                                          LinearQuantizer_process)
from mqbench.utils.logger import logger


class STPU_process(LinearQuantizer_process):

    def remove_fakequantize_and_collect_params(self, onnx_path, model_name):
        model = onnx.load(onnx_path)
        graph = model.graph
        out2node, inp2node = update_inp2node_out2node(graph)
        name2data = prepare_data(graph)
        named_initializer = prepare_initializer(graph)

        out2node, inp2node = update_inp2node_out2node(graph)

        quant_params = OrderedDict()
        nodes_to_be_removed = []
        for node in graph.node:
            if node.op_type in PERTENSOR_FAKEQUANTIZER:
                nodes_to_be_removed.append(node)
                nodes_to_be_removed.extend(get_constant_inputs(node, out2node))

                if node.output[0] not in inp2node:
                    assert node.output[0] in [x.name for x in graph.output]
                    inp2node[node.output[0]] = []

                next_nodes = inp2node[node.output[0]]
                if len(next_nodes) == 1 and next_nodes[0][1] == 1 and next_nodes[0][0].op_type in ['Gemm', 'Conv']:
                    # fake quantize for weights
                    redundant_nodes = self.deal_with_weight_fakequant(node, out2node, inp2node, named_initializer)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    nodes_to_be_removed.extend(redundant_nodes)
                    self.clip_weight(node, name2data, inp2node, named_initializer)
                    # [-127 * scale, 127 * scale]
                    quant_params[next_nodes[0][0].name + '_weights'] = {
                        "min": -127 * scale,
                        "max": 127 * scale
                    }
                else:
                    # fake quantize for activations
                    self.deal_with_activation_fakequant(node, inp2node)
                    tensor_name, scale, zero_point, qmin, qmax = self.parse_qparams(node, name2data)
                    for out in graph.output:
                        if out.name == node.output[0]:
                            out.name = tensor_name
                    quant_params[tensor_name] = {
                        "min": -127 * scale,
                        "max": 127 * scale
                    }
        # Update bias scale = input scale * weight scale
        for node in graph.node:
            if node.op_type in ['Gemm', 'Conv'] and len(node.input) == 3:
                walpha = (quant_params[node.name + '_weights']['max'] - quant_params[node.name + '_weights']['min']) \
                    / (2 ** 8 - 2)
                ialpha = (quant_params[node.input[0]]['max'] - quant_params[node.input[0]]['min']) / (2 ** 8 - 2)
                quant_params[node.name + '_bias'] = {
                    "alpha": walpha * ialpha,
                    "zero_point": 0
                }
            if node.op_type == 'Sigmoid':
                quant_params[node.output[0]] = {
                    "min": -1,
                    "max": 1
                }
        quant_params = self.post_process_clip_ranges(quant_params, graph, inp2node)
        self.merge_relu_layer(graph, quant_params, out2node)
        for node in graph.node:
            self.update_emin(node, quant_params, named_initializer)
        # Delete node and init.
        for node in nodes_to_be_removed:
            graph.node.remove(node)
        named_initializer = prepare_initializer(graph)
        for name, initial_data in named_initializer.items():
            if name in (out2node.keys() | inp2node.keys()):
                continue
            graph.initializer.remove(initial_data)
        # Save range file and model.
        output_path = os.path.dirname(onnx_path)
        context_filename = os.path.join(output_path, f"{model_name}_stpu_minmax.json")
        with open(context_filename, "w") as f:
            json.dump(quant_params, f, indent=4)
        onnx_filename = os.path.join(output_path, f"{model_name}_for_stpu.onnx")
        onnx.save(model, onnx_filename)

        logger.info("Finish deploy process.")

    def merge_relu_layer(self, graph, quant_params, out2node):
        for node in graph.node:
            if node.op_type == 'Relu' and node.output[0] in quant_params:
                prev_nodes = out2node[node.input[0]]
                quant_params[prev_nodes.output[0]] = quant_params[node.output[0]].copy()
                logger.info("Merge conv + relu range pass {} to {}".format(node.output[0], prev_nodes.output[0]))

    def update_emin(self, node, quant_params, named_initializer):
        '''EMIN is some kind of magic number for STPU.
        Do not try to understand it.
        '''
        def find_e(v):
            v_ = abs(v)
            if v_ == 0:
                return 0

            for e in range(1, 254):
                r_e = e - 127
                if (v >= 2 ** r_e) and (v < 2 ** (r_e + 1)):
                    return e

            if v < 2 ** (-126):
                return 1
            return 254

        def find_interp_emin(vmax, r):
            emax = find_e(vmax)
            return emax - (22 - r)

        def find_conv_emin(i_vmax, w_vmax, o_vmax, n, r):
            n = n ** .5
            vmax = max(n * i_vmax * w_vmax, o_vmax)
            emax = find_e(vmax)
            return emax - (12 - r)

        if node.op_type in ['Upsample', 'DynamicUpsample']:
            emin = find_interp_emin(quant_params[node.output[0]]['max'], 2)
            quant_params[node.output[0]]['emin'] = emin          
        if node.op_type in ['Conv', 'ConvTranspose']:
            weight_shape = named_initializer[node.input[1]].dims
            n = weight_shape[1] * weight_shape[2] * weight_shape[3]
            i_vmax = quant_params[node.input[0]]['max']
            o_vmax = quant_params[node.output[0]]['max']
            w_vmax = quant_params[node.name + '_weights']['max']
            emin = find_conv_emin(i_vmax, w_vmax, o_vmax, n, 2)
            quant_params[node.output[0]]['emin'] = emin
        if node.op_type in ['Gemm']:
            # We fix it here.
            quant_params[node.output[0]]['emin'] = 128

remove_fakequantize_and_collect_params_stpu = STPU_process().remove_fakequantize_and_collect_params
