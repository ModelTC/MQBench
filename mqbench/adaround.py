import numpy as np
from torch.fx import GraphModule, Node
from torch import fx, nn
from torch.nn import Module
import torch

USE_LINK = False
USE_DDP = False

try:
    import spring.linklink as link
    assert link.is_initialized()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True

import numpy as np
from typing import List

from mqbench.utils.logger import logger
from mqbench.utils.hook import DataSaverHook, StopForwardException
from mqbench.utils import deepcopy_graphmodule
from mqbench.utils.state import enable_quantization, disable_all

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)


def lp_loss(pred, tgt, p=2.0):
    """
    loss function measured in L_p Norm
    """
    return (pred - tgt).abs().pow(p).sum(1).mean()


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
                _ = model(batch.to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append([inp.detach() for inp in inp_saver.input_store])
                else:
                    cached[0].append([inp.detach().cpu() for inp in inp_saver.input_store])  # tuple/list one
            if store_oup:
                if keep_gpu:
                    cached[1].append(oup_saver.output_store.detach())
                else:
                    cached[1].append(oup_saver.output_store.detach().cpu())
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

    assert USE_LINK or USE_DDP, 'either USE_LINK or USE_DDP should be True'
    world_size = link.get_world_size() if USE_LINK else dist.get_world_size()

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
            cur_inp = [inp.to(device) for inp in cached_inps[0][idx]]
            cur_sym = [sym.to(device) for sym in cached_inps[1][idx]]
            cur_inp = [torch.where(torch.rand_like(inp) < config.prob, inp, sym) for inp, sym in zip(cur_inp, cur_sym)]
        else:
            cur_inp = cached_inps[idx]
            cur_inp = [inp.to(device) for inp in cur_inp]
        cur_out = cached_oups[idx].to(device)
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
        if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, torch.quantization.FakeQuantizeBase) and 'post_act_fake_quantize' in name:
            layer.prob = 1.0   # recover to promise that drop activation quantization only occurs at reconstruction phase


def extract_subgraph(orig_module: nn.Module, nodes: List[fx.Node], inputs: List[fx.Node], output: fx.Node):
    """
    Given lists of nodes from an existing graph that represent a subgraph, returns a submodule that executes that subgraph.
    """
    new_graph = fx.Graph()
    env = dict()
    for input in inputs:
        new_node = new_graph.placeholder(input.name)
        env[input] = new_node
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


def adaround(model, cali_data, config):
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
            cur_node = layer_node_list[-1]
            fp32_module = fp32_modules[cur_node.target]
            fp32_inp_module = fp32_modules[node.target]
            quant_module = quant_modules[node.target]
            fp32_inps, fp32_oups = save_inp_oup_data(fp32_model, fp32_inp_module, fp32_module, cali_data,
                                                     store_inp=(config.prob < 1.0), store_oup=True, keep_gpu=config.keep_gpu)
            quant_inps, _ = save_inp_oup_data(quant_model, quant_module, None, cali_data, store_inp=True,
                                              store_oup=False, keep_gpu=config.keep_gpu)
            logger.info('the node list is below!')
            logger.info(layer_node_list)
            subgraph = extract_subgraph(quant_modules, layer_node_list, node.all_input_nodes, cur_node)
            logger.info(subgraph)
            cached_inps = (quant_inps, fp32_inps) if config.prob < 1.0 else quant_inps
            cached_oups = fp32_oups
            subgraph_reconstruction(subgraph, cached_inps, cached_oups, config)
            for x in layer_node_list:
                checked_nodes[x] = True
    return quant_model
