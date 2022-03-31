import torch
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module

USE_LINK = False
USE_DDP = False

__all__ = ['ptq_reconstruction']

try:
    import spring.linklink as link
    if not link.is_initialized():
        link.initialize()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True

import numpy as np
from typing import List

from mqbench.utils.logger import logger
from mqbench.utils.hook import DataSaveHookByNode
from mqbench.utils import deepcopy_graphmodule
from mqbench.utils.state import enable_quantization, disable_all

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)


def lp_loss(pred, tgt, p=2.0):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


def to_device(data, device='cpu'):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    else:
        return data


def tensor_detach(data):
    if isinstance(data, torch.Tensor):
        return data.detach()
    elif isinstance(data, dict):
        for key in data:
            data[key] = tensor_detach(data[key])
        return data 
    else:
        return data 


def save_inps_data_by_node(model, modules, inp_nodes, out_node, node2idx, cali_data, keep_gpu):
    # return a tuple of two list
    # return a batch x args input and batch output 
    # provides each node's output
    for inp in inp_nodes:
        assert node2idx[inp] < node2idx[out_node], 'topology error'
    device = next(model.parameters()).device
    inp_savers = []
    handles = []
    for inp in inp_nodes:
        module = modules[inp.target]
        inp_saver = DataSaveHookByNode(None, inp, False, True)
        handle = module.register_forward_hook(inp_saver)
        inp_savers.append(inp_saver)
        handles.append(handle)
    handles.append(handle)
    cached_in = []
    with torch.no_grad():
        for batch in cali_data:
            cache_inp_dict = {}
            model(to_device(batch, device))
            for inp_saver in inp_savers:
                cache_inp_dict.update(inp_saver.output_store)
            if not keep_gpu:
                cache_inp_dict = to_device(cache_inp_dict, 'cpu')
            cached_in.append(cache_inp_dict)
    for handle in handles:
        handle.remove()
    torch.cuda.empty_cache()
    return cached_in

def save_oup_data_by_node(subgraph, cached_fp_inps, keep_gpu):
    device = next(subgraph.parameters()).device
    cached_out = []
    with torch.no_grad():
        for batch in cached_fp_inps:
            cur_fp_inp = to_device(batch, device)
            cur_inp_dict = {}
            for node in cur_fp_inp:
                cur_inp_dict[node.name] = cur_fp_inp[node]
            cur_fp_out = subgraph(**cur_inp_dict)
            if not keep_gpu:
                cur_fp_inp = to_device(cur_fp_inp, 'cpu')
            cached_out.append(cur_fp_out)
    return cached_out


class LinearTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class CosineTempDecay:
    def __init__(self, t_max=10000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 subgraph: Module,
                 weight: float = 1.,
                 max_count: int = 10000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.):

        self.subgraph = subgraph
        self.weight = weight
        self.loss_start = max_count * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, warm_up=warm_up,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :return: total loss function
        """
        self.count += 1
        rec_loss = lp_loss(pred, tgt, p=self.p)

        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0
        else:
            round_loss = 0
            for layer in self.subgraph.modules():
                if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


def _flatten_args(node):
    flattned_args = []
    if isinstance(node, dict):
        for v in node.values():
            flattned_args.extend(_flatten_args(v))
    elif isinstance(node, tuple) or isinstance(node, list):
        for n in node:
            flattned_args.extend(_flatten_args(n))
    else:
        flattned_args.extend([node])
    return flattned_args


def _drop_input(q, fp, rate):
    if isinstance(q, torch.Tensor):
        return torch.where(torch.rand_like(q) < rate, q, fp)
    elif isinstance(q, dict):
        out = {}
        for k in q:
            out[k] = _drop_input(q[k], fp[k], rate)
        return out
    elif isinstance(q, list):
        out = []
        for idx in range(len(q)):
            out.append(_drop_input(q[idx], fp[idx], rate))
        return out 
    elif isinstance(q, tuple):
        out = []
        for idx in range(len(q)):
            out.append(_drop_input(q[idx], fp[idx], rate))
        return tuple(out)
    else:
        raise ValueError('Data type not supported.')

def append_extra_inputs(nodes, layer_node_list):
    # there are some nodes in the block which are used but not in the list. 
    # e.g. a global dict used in UP or EOD.
    extra_inputs = [] 
    for node in layer_node_list:
        for arg in _flatten_args(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg not in layer_node_list:
                    extra_inputs.append(arg)
    return extra_inputs


def find_cur_node(layer_node_list):
    for node in reversed(layer_node_list):
        if node.target == 'update':
            continue 
        if isinstance(node.target, str) and 'const' in node.target:
            continue 
        return node
    raise ValueError('Bad layer node list provided.')

def subgraph_reconstruction(subgraph, placeholders, cached_inps, cached_oups, config):
    global USE_LINK
    global USE_DDP
    device = next(subgraph.parameters()).device
    w_para, a_para = [], []
    w_opt, w_scheduler = None, None
    if hasattr(config, 'scale_lr'):
        a_para = []
    for name, layer in subgraph.named_modules():
        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            weight_quantizer = layer.weight_fake_quant
            weight_quantizer.init(layer.weight.data, config.round_mode)
            w_para += [weight_quantizer.alpha]
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and 'post_act_fake_quantize' in name:
            if hasattr(config, 'scale_lr'):
                logger.info('learn the scale for {}'.format(name))
                a_para += [layer.scale]
            layer.prob = config.prob
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=config.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(a_opt, T_max=config.max_count, eta_min=0.)
    else:
        a_opt, a_scheduler = None, None
    w_opt = torch.optim.Adam(w_para)

    loss_func = LossFunction(subgraph=subgraph, weight=config.weight, max_count=config.max_count, b_range=config.b_range,
                             warm_up=config.warm_up)

    if USE_LINK:
        world_size = link.get_world_size()
    elif USE_DDP:
        world_size = dist.get_world_size()
    else:
        world_size = 1

    logger.info('The world size is {}.'.format(world_size))
    '''start training'''
    logger.info('start tuning by adaround')
    if config.prob < 1.0:
        sz = len(cached_inps[0])
    else:
        sz = len(cached_inps)
    for i in range(config.max_count):
        idx = np.random.randint(0, sz)
        if config.prob < 1.0:
            cur_inp = to_device(cached_inps[0][idx], device)
            cur_sym = to_device(cached_inps[1][idx], device)
            cur_inp = _drop_input(cur_inp, cur_sym, config.prob)
        else:
            cur_inp = cached_inps[idx]
            cur_inp = to_device(cur_inp, device)
        cur_out = to_device(cached_oups[idx], device)
        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        cur_inp_dict = {} 
        for node in cur_inp:
            cur_inp_dict[node.name] = cur_inp[node]
        cur_out_tensor = cur_out
        out_quant = subgraph(**cur_inp_dict)
        err = loss_func(out_quant, cur_out_tensor)
        err /= world_size
        err.backward()
        if world_size > 1:
            for param in w_para:
                if USE_LINK:
                    link.allreduce(param.grad.data)
                elif USE_DDP:
                    dist.all_reduce(param.grad.data)
        w_opt.step()
        if a_opt:
            a_opt.step()
        if w_scheduler:
            w_scheduler.step()
        if a_scheduler:
            a_scheduler.step()
    torch.cuda.empty_cache()
    for name, layer in subgraph.named_modules():
        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and 'post_act_fake_quantize' in name:
            layer.prob = 1.0   # recover to promise that drop activation quantization only occurs at reconstruction phase




def extract_subgraph_with_placeholder(orig_module: nn.Module, nodes: List[fx.Node], inputs: List[fx.Node], output: fx.Node, extra_inputs: List[fx.Node]):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env = dict()
    placeholders = []
    for input in set(inputs + extra_inputs):
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
        placeholders.append(input)
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    # create this or there will not be return value
    new_graph.output(env[output])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph), placeholders


def find_num_nodes(nodes):
    num = 0
    for node in nodes:
        if isinstance(node, Node):
            num += 1
    return num


# Recommend: log this to check if the layer is right. You can define your own layer manually or automatically like this
# extract the linked-list/single-chain
def extract_layer(node, fp32_modules):
    layer_node_list = []
    cur_node = node
    is_next_block = False  # check whether stoped by a block
    while True:
        logger.debug('cur_node in layer is {}'.format(cur_node))
        layer_node_list.append(cur_node)  # valid node here
        stop = (len(cur_node.users) == 0)
        for user in cur_node.users:
            if user.op == 'call_module' and isinstance(fp32_modules[user.target], _ADAROUND_SUPPORT_TYPE):
                stop = True
            # TODO: only short-cut here, consider more here
            # TODO: can also use un/completed to check here.
            if ('add' in user.name and user.op in ['call_function', 'call_method']):
                stop = True
            if user.op == 'output':
                is_next_block, stop = True, True
        if stop:
            break
        cur_node = list(cur_node.users.keys())[0]
    if find_num_nodes(cur_node.users) > 1:
        is_next_block = True
    return layer_node_list, is_next_block


# Recommend: log this to check if the block is right. You can define your own block manually or automatically like this
# extract the block one such as short-cut
def extract_block(input_nodes, fp32_modules, depth=0):
    if depth > 2:
        # stack 2 or 3 layers for no short-cut structure
        return []
    layer_node_list = []
    is_block = False
    cnt = dict()
    cur_node = None
    q, p = [], []  # q records the completed node, p records the uncompleted nodes
    for input in input_nodes:
        for user in input.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
    while len(q) != 0:
        cur_node = q.pop(0)  # valid node here
        logger.debug('cur node is {}'.format(cur_node))
        if len(p) == 0 and len(q) == 0:
            break
        layer_node_list.append(cur_node)
        for user in cur_node.users:
            if user not in cnt:
                cnt[user] = find_num_nodes(user.args)
                if cnt[user] > 1:
                    is_block = True
                p.append(user)
            cnt[user] -= 1
            if cnt[user] == 0:
                q.append(user)
                p.remove(user)
        logger.debug('uncompleted nodes are {}'.format(p))
    if not cur_node:
        return layer_node_list
    if cur_node.target == 'update':
        if len(q) > 0:
            cur_node = q.pop(0)
        else:
            exp_nodes = []
            is_next_block = False
    exp_nodes, is_next_block = extract_layer(cur_node, fp32_modules)
    if is_block or is_next_block:
        return layer_node_list + exp_nodes
    else:
        return layer_node_list + exp_nodes + extract_block([exp_nodes[-1]], fp32_modules, depth + 1)



def _get_items_for_the_graph(model: GraphModule) -> dict:
    def _update_getitem_path(getitem_args_dict):
        for node in getitem_args_dict:
            args_list = getitem_args_dict[node]
            while args_list[0] in getitem_args_dict:
                args_list = getitem_args_dict[args_list[0]] + args_list[1:]
            getitem_args_dict[node] = args_list
        return getitem_args_dict

    def _getitem_from_args(args, original_args_dict):
        ret = original_args_dict
        for a in args:
            try:
                ret = ret[a]
            except (IndexError, KeyError):
                return {}
        return ret 
    import operator
    nodes = list(model.graph.nodes)
    # the getitem's call graph
    getitem_args_dict = {}
    # the dict used in the model 
    original_key_dict = {}
    getitem2node = {}
    for node in nodes:
        # update the getitems
        if node.target == operator.getitem:
            getitem_args_dict[node] = list(node.args)
            getitem_args_dict = _update_getitem_path(getitem_args_dict)
        elif node.target == 'update':
            if node.args[0] not in original_key_dict:
                original_key_dict[node.args[0]] = {}
            original_key_dict[node.args[0]].update(node.args[1])
    for node in getitem_args_dict:
        val = _getitem_from_args(getitem_args_dict[node], original_key_dict)
        if isinstance(val, torch.fx.node.Node):
            getitem2node[node] = val
    node2idx = {}
    for idx, node in enumerate(model.graph.nodes):
        node2idx[node] = idx 

    return getitem2node, node2idx


def replace_getitems(layer_node_list, getitem2node):
    for idx, _ in enumerate(layer_node_list):
        if layer_node_list[idx] in getitem2node:
            layer_node_list[idx] = getitem2node[layer_node_list[idx]]
    return layer_node_list 



def remove_non_tensor_nodes(node_list):
    remove_list = []
    for node in node_list:
        if node.target == 'update':
            if len(node.args[0].users) == 1:
                remove_list.append(node.args[0])
            remove_list.append(node)
    return [node for node in node_list if node not in remove_list]


def nodes_from_name(nodes, special_block, prefix):
    input_nodes = []
    for node in nodes:
        if node.name in special_block[prefix]:
            input_nodes.append(node)
    return input_nodes 

def ptq_reconstruction(model: GraphModule, cali_data: list, config: dict):
    r"""
    Reconsturction for AdaRound, BRECQ, QDrop.
    Basic optimization objective:

    .. math::

        \mathop{\arg\min}_{\mathbf{V}}\ \ || Wx-\tilde{W}x ||_F^2 + \lambda f_{reg}(\mathbf{V}),

        \tilde{W}=s \cdot clip\left( \left\lfloor\dfrac{W}{s}\right\rfloor+h(\mathbf{V}), n, p \right)

    where :math:`h(\mathbf{V}_{i,j})=clip(\sigma(\mathbf{V}_{i,j})(\zeta-\gamma)+\gamma, 0, 1)`, and :math:`f_{reg}(\mathbf{V})=\mathop{\sum}_{i,j}{1-|2h(\mathbf{V}_{i,j})-1|^\beta}`. By annealing on :math:`\beta`, the rounding mask can adapt freely in initial phase and converge to 0 or 1 in later phase.

    Args:
        model (torch.nn.Module): a prepared GraphModule to do PTQ
        cali_data (List): a list of calibration tensor
        config (dict): a config for PTQ reconstruction

    >>> sample config : {
            pattern: block (str, Available options are [layer, block].)
            scale_lr: 4.0e-5 (learning rate for learning step size of activation)
            warm_up: 0.2 (0.2 * max_count iters without regularization to floor or ceil)
            weight: 0.01 (loss weight for regularization item)
            max_count: 20000 (optimization iteration)
            b_range: [20,2] (beta decaying range )
            keep_gpu: True (calibration data restore in gpu or cpu)
            round_mode: learned_hard_sigmoid (ways to reconstruct the weight, currently only support learned_hard_sigmoid)
            prob: 0.5 (dropping probability of QDROP)
        }

    """
    # assert model is on cuda
    if not config.keep_gpu:
        cali_data = [inp.cpu() for inp in cali_data]
    '''set state first'''

    fp32_model = model
    fp32_model.eval()
    quant_model = deepcopy_graphmodule(model)
    quant_model.eval()
    disable_all(fp32_model)
    enable_quantization(quant_model)
    torch.cuda.empty_cache()
    nodes = list(quant_model.graph.nodes)
    fp32_modules = dict(fp32_model.named_modules())
    quant_modules = dict(quant_model.named_modules())
    checked_nodes = dict()
    getitem2node, node2idx = _get_items_for_the_graph(quant_model)
    for node in nodes:
        if node in checked_nodes:
            continue
        if node.op == "call_module" and isinstance(fp32_modules[node.target], _ADAROUND_SUPPORT_TYPE):
            logger.info('prepare {} reconstruction for {}'.format(config.pattern, node.target))
            input_nodes = node.all_input_nodes
            if config.pattern == 'layer':
                layer_node_list, _ = extract_layer(node, fp32_modules)
            elif config.pattern == 'block':
                layer_node_list = extract_block(input_nodes, fp32_modules)
            else:
                raise NotImplementedError
            layer_node_list = replace_getitems(layer_node_list, getitem2node)
            layer_node_list = sorted(layer_node_list, key=lambda x: node2idx[x])
            layer_node_list = remove_non_tensor_nodes(layer_node_list)
            cur_node = find_cur_node(layer_node_list)
            logger.info('the node list is below!')
            logger.info(layer_node_list)
            extra_inputs = append_extra_inputs(nodes, layer_node_list)
            subgraph, placeholders = extract_subgraph_with_placeholder(quant_modules, layer_node_list, input_nodes, cur_node, extra_inputs)
            logger.info(subgraph)
            fp32_inps = save_inps_data_by_node(fp32_model, fp32_modules, placeholders, cur_node, node2idx, cali_data, config.keep_gpu)
            quant_inps = save_inps_data_by_node(quant_model, quant_modules, placeholders, cur_node, node2idx, cali_data, config.keep_gpu)
            fp32_oups = save_oup_data_by_node(subgraph, fp32_inps, config.keep_gpu)
            cached_inps = (quant_inps, fp32_inps) if config.prob < 1.0 else quant_inps
            cached_oups = fp32_oups
            subgraph_reconstruction(subgraph, placeholders, cached_inps, cached_oups, config)
            for x in layer_node_list:
                checked_nodes[x] = True
    return quant_model
