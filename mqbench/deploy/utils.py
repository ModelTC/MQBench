import copy
import onnx
import numpy as np
from onnx import numpy_helper
from onnx import TensorProto


class ONNXGraph(object):
    def __init__(self, onnx_model_path):
        '''Describe onnx graph
        args: 
            input_map[tensor_name] = node which input is tensor_name
            output_map[tensor_name] = node which output is tensor_name
        '''
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        self.initializer = {}
        self.input_map = {}
        self.output_map = {}
        self.topologize_graph()
        self.prepare_initializer()

    def prepare_initializer(self):
        self.initializer.clear()
        for idx, init in enumerate(self.graph.initializer):
            self.initializer[init.name] = (init, idx)

    def get_constant(self, name):
        for node in self.model.graph.node:
            if node.op_type == 'Constant':
                if node.output[0] == name:
                    return numpy_helper.to_array(node.attribute[0].t).tolist()

    def get_initializer(self, initializer_name):
        return numpy_helper.to_array(self.initializer[initializer_name][0])

    def set_initializer(self, initializer_name, value_tensor, raw=True):
        idx = None
        if initializer_name in self.initializer:
            idx = self.initializer[initializer_name][1]
        if raw:
            initializer = numpy_helper.from_array(value_tensor)
        else:
            if value_tensor.dtype == np.float32:
                data_type = TensorProto.FLOAT
            if value_tensor.dtype == np.uint8:
                data_type = TensorProto.UINT8
            initializer = onnx.helper.make_tensor(name=initializer_name, 
                                                  data_type=data_type, 
                                                  dims=[], 
                                                  vals=value_tensor, 
                                                  raw=False)
        initializer.name = initializer_name
        if idx is not None:
            self.graph.initializer.remove(self.graph.initializer[idx])
        self.graph.initializer.append(initializer)
        self.prepare_initializer()

    def topologize_graph(self):
        self.input_map.clear()
        self.output_map.clear()
        for node in self.graph.node:
            for output_name in node.output:
                self.output_map[output_name] = node
            for input_name in node.input:
                if input_name not in self.input_map:
                    self.input_map[input_name] = []
                self.input_map[input_name].append(node)

    def get_tensor_producer(self, output_name):
        if output_name not in self.output_map:
            return 'INPUT_TOKEN'
        return self.output_map[output_name]

    def get_tensor_consumer(self, input_name):
        if input_name not in self.input_map:
            return ['OUTPUT_TOKEN']
        return self.input_map[input_name]

    def save_onnx_model(self, model_path):
        onnx.save(self.model, model_path)

    def remove_node_purely(self, node):
        self.graph.node.remove(node)

    def insert_node_purely(self, node, idx=0):
        self.graph.node.insert(idx, node)

    def del_initializer(self, initializer_name):
        if initializer_name in self.initializer:
            del(self.initializer[initializer_name])

    def optimize_model(self):
        # Delete redundant nodes.
        remove_node_list = []
        for node in self.model.graph.node:
            if len(node.input) == 0:
                not_be_used = True
                for output_name in node.output:
                    if output_name in self.input_map:
                        not_be_used = False
                        break
                if not_be_used:
                    remove_node_list.append(node)
        for node in remove_node_list:
            self.remove_node_purely(node)
        self.topologize_graph()
        # Delete redundant initializers.
        for initializer_name in self.initializer:
            if initializer_name not in self.input_map:
                self.del_initializer(initializer_name)
        # Make node in topology order.
        exist_input = [input_node.name for input_node in self.model.graph.input]
        origin_node_num = len(self.model.graph.node)
        finished_node_name = []
        # O(n^2)
        while len(finished_node_name) < origin_node_num:
            node_detect = False
            for i in range(origin_node_num):
                node = self.model.graph.node[i]
                all_inputs_exist = True
                for input_name in node.input:
                    if input_name not in exist_input and input_name not in self.initializer:
                        all_inputs_exist = False
                        break
                if all_inputs_exist:
                    if node.name not in finished_node_name:
                        node_detect = True
                        finished_node_name.append(node.name)
                        self.model.graph.node.append(node)
                        for output_name in node.output:
                            exist_input.append(output_name)
            assert node_detect, "Graph is illegel, error occured!"
        for i in range(origin_node_num):
            self.model.graph.node.remove(self.model.graph.node[0])

    def set_opset_version(self, domain, version):
        opset_info = copy.deepcopy(self.model.opset_import[0])
        opset_info.domain = domain
        opset_info.version = version
        self.model.opset_import.insert(0, opset_info)