import torch
import torch.nn.intrinsic.qat as nniqat
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
from mqbench.utils.hook import DataSaverHook, StopForwardException
from mqbench.utils import deepcopy_graphmodule, topology_order
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
    elif isinstance(data, list):
        for idx, _ in enumerate(data):
            data[idx] = to_device(data[idx], device)
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


def save_inp_oup_data(model: GraphModule, inp_module: Module, oup_module: Module, cali_data: list, store_inp=True, store_oup=True,
                      keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    :param fp_model: fp_model
    :param quant_model: quant_model
    :param cali_data: calibration data set
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    if store_inp:
        assert inp_module is not None
        inp_saver = DataSaverHook(store_input=store_inp, store_output=False, stop_forward=(not store_oup))
        inp_handle = inp_module.register_forward_hook(inp_saver)
    if store_oup:
        assert oup_module is not None
        oup_saver = DataSaverHook(store_input=False, store_output=store_oup, stop_forward=True)
        oup_handle = oup_module.register_forward_hook(oup_saver)
    cached = ([], [])
    with torch.no_grad():
        for batch in cali_data:
            try:
                _ = model(to_device(batch, device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append([tensor_detach(inp) for inp in inp_saver.input_store])
                else:
                    cached[0].append([to_device(tensor_detach(inp), 'cpu') for inp in inp_saver.input_store])  # tuple/list one
            if store_oup:
                if keep_gpu:
                    cached[1].append(tensor_detach(oup_saver.output_store))
                else:
                    cached[1].append(to_device(tensor_detach(oup_saver.output_store), 'cpu'))
    if store_inp:
        inp_handle.remove()
    if store_oup:
        oup_handle.remove()
    torch.cuda.empty_cache()
    return cached


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



def find_cur_node(layer_node_list):
    for node in reversed(layer_node_list):
        if node.target == 'update':
            continue 
        if isinstance(node.target, str) and 'const' in node.target:
            continue 
        return node
    raise ValueError('Bad layer node list provided.')

def subgraph_reconstruction(subgraph, cached_inps, cached_oups, config):
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
            # assert isinstance(weight_quantizer, adaround_quantizer) is True
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

    if any([USE_DDP, USE_LINK]):
        world_size = link.get_world_size() if USE_LINK else dist.get_world_size()
    else:
        world_size = 1

    logger.info('The world size is {}.'.format(world_size))
    '''start training'''
    logger.info('start tuning by adaround')
    if config.prob < 1.0:
        # cache inps: drop x args x batch x data
        sz = len(cached_inps[0][0])
        num_args = len(cached_inps[0])
    else:
        # cache inps: args x batch x data
        sz = len(cached_inps[0])
        num_args = len(cached_inps)
    for i in range(config.max_count):
        idx = np.random.randint(0, sz)
        cur_args = []
        for a in range(num_args):
            if config.prob < 1.0:
                cur_inp = to_device(cached_inps[0][a][idx], device)
                cur_sym = to_device(cached_inps[1][a][idx], device)
                cur_inp = torch.where(torch.rand_like(cur_inp) < config.prob, cur_inp, cur_sym)
            else:
                cur_inp = to_device(cached_inps[a][idx], device)
            cur_args.append(cur_inp)
        cur_args = tuple(cur_args)
        cur_out = to_device(cached_oups[idx], device)
        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()
        out_quant = subgraph(*cur_inp)
        err = loss_func(out_quant, cur_out)
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
        if isinstance(layer, (nniqat.ConvBnReLU2d, nniqat.ConvBn2d)):
            # We need to do bn fold simulation here.
            weight_quantizer = layer.weight_fake_quant
            scale_factor = layer.bn.weight / torch.sqrt(layer.bn.running_var + layer.bn.eps)
            merged_rounded_weight = weight_quantizer.get_hard_value(
                layer.weight.data * scale_factor.reshape([-1] + [1] * (len(layer.weight.shape) - 1)))
            layer.weight.data = merged_rounded_weight / scale_factor.reshape([-1] + [1] * (len(merged_rounded_weight.shape) - 1))
            weight_quantizer.adaround = False
        elif isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            assert not hasattr(layer, 'bn'), 'Layer {} has BN ! Should not reach here.'.format(name)
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and 'post_act_fake_quantize' in name:
            layer.prob = 1.0   # recover to promise that drop activation quantization only occurs at reconstruction phase


def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], output: fx.Node):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env = dict()
    for node in nodes:
        for arg in _flatten_args(node.args):
            if isinstance(arg, torch.fx.Node):
                if arg not in nodes:
                    new_node = new_graph.placeholder(arg.name)
                    env[arg] = new_node
    for node in nodes:
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    # create this or there will not be return value
    new_graph.output(env[output])
    new_graph.lint()
    return fx.GraphModule(orig_module, new_graph)


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
    exp_nodes, is_next_block = extract_layer(cur_node, fp32_modules)
    if is_block or is_next_block:
        return layer_node_list + exp_nodes
    else:
        return layer_node_list + exp_nodes + extract_block([exp_nodes[-1]], fp32_modules, depth + 1)


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
    for node in nodes:
        if node in checked_nodes:
            continue
        if node.op == "call_module" and isinstance(fp32_modules[node.target], _ADAROUND_SUPPORT_TYPE):
            logger.info('prepare {} reconstruction for {}'.format(config.pattern, node.target))
            if config.pattern == 'layer':
                layer_node_list, _ = extract_layer(node, fp32_modules)
            elif config.pattern == 'block':
                layer_node_list = extract_block(node.all_input_nodes, fp32_modules)
            else:
                raise NotImplementedError
            missing_inputs = [] 
            for node in layer_node_list:
                for arg in _flatten_args(node.args):
                    if isinstance(arg, torch.fx.Node):
                        if arg not in layer_node_list:
                            missing_inputs.append(arg)
            layer_node_list.extend(missing_inputs)
            layer_node_list = sorted(layer_node_list, key=lambda x: topology_order(quant_model)[x])
            logger.info('the node list is below!')
            logger.info(layer_node_list)
            cur_node = find_cur_node(layer_node_list)
            fp32_module = fp32_modules[cur_node.target]
            fp32_all_inps = []
            quant_all_inps = []
            fp32_final_oups = None
            out_is_cached = False
            for _node in layer_node_list:
                if all([arg in layer_node_list for arg in _flatten_args(_node.args) if isinstance(arg, torch.fx.Node)]):
                    continue
                else:
                    fp32_inp_module = fp32_modules[_node.target]
                    quant_module = quant_modules[_node.target]
                    # fp32 inps: [out_b1, out_b2, ...]
                    _, fp32_inps = save_inp_oup_data(fp32_model, None, fp32_inp_module, cali_data,
                                                     store_inp=False, store_oup=(config.prob < 1.0), keep_gpu=config.keep_gpu)
                    _, fp32_oups = save_inp_oup_data(fp32_model, None, fp32_module, cali_data,
                                                     store_inp=False, store_oup=(not out_is_cached), keep_gpu=config.keep_gpu)
                    _, quant_inps = save_inp_oup_data(quant_model, None, quant_module, cali_data, store_inp=False,
                                                      store_oup=True, keep_gpu=config.keep_gpu)
                    fp32_all_inps.append(fp32_inps)
                    quant_all_inps.append(quant_all_inps)
                    if not out_is_cached:
                        fp32_final_oups = fp32_oups
                        out_is_cached = True
            cached_inps = (quant_inps, fp32_inps) if config.prob < 1.0 else quant_inps
            cached_oups = fp32_final_oups
            subgraph = extract_subgraph(quant_modules, layer_node_list, cur_node)
            logger.info(subgraph)
            subgraph_reconstruction(subgraph, cached_inps, cached_oups, config)
            for x in layer_node_list:
                checked_nodes[x] = True
    return quant_model
