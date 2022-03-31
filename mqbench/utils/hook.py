class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass
class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException

class DataSaveHookByNode:
    """
    Forward hook that store a dict with the node as key.
    """
    def __init__(self, input_node, output_node, store_input, store_output):
        self.input_node = input_node
        self.output_node = output_node 
        self.store_input = store_input
        self.store_output = store_output 
        self.input_store = {}
        self.output_store = {}

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store.update({self.input_node: input_batch})
        if self.store_output:
            self.output_store.update({self.output_node: output_batch})