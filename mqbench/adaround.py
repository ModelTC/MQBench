from torch.fx import GraphModule
from torch.nn import Module
import torch
import spring.linklink as link
import numpy as np

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


def save_inp_oup_data(model: GraphModule, module: Module, cali_data: list, store_inp=True, store_oup=True,
                      keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.
    :param fp_model: fp_model
    :param quant_model: quant_model
    :param cali_data: calibration data set
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    assert (not store_inp or not store_oup)
    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_input=store_inp, store_output=store_oup, stop_forward=True)
    handle = module.register_forward_hook(data_saver)
    cached = []
    with torch.no_grad():
        for batch in cali_data:
            try:
                _ = model(batch.to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached.append([inp.detach() for inp in data_saver.input_store])
                else:
                    cached.append([inp.detach().cpu() for inp in data_saver.input_store])  # tuple/list one
            if store_oup:
                if keep_gpu:
                    cached.append(data_saver.output_store.detach())
                else:
                    cached.append(data_saver.output_store.detach().cpu())
    handle.remove()
    torch.cuda.empty_cache()
    return cached


# TODO: add linear one
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
            # return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
            return self.end_b + 0.5 * (self.start_b - self.end_b) * (1 + np.cos(rel_t * np.pi))


class LossFunction:
    r'''loss function to calculate mse reconstruction loss and relaxation loss
    use some tempdecay to balance the two losses.
    '''

    def __init__(self,
                 layer: Module,
                 weight: float = 1.,
                 max_count: int = 10000,
                 b_range: tuple = (20, 2),
                 warm_up: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.weight = weight
        self.loss_start = max_count * warm_up
        self.p = p

        self.temp_decay = CosineTempDecay(max_count, warm_up=warm_up,
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
            round_vals = self.layer.weight_fake_quant.rectified_sigmoid()
            round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        if self.count % 500 == 0:
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


def layer_reconstruction(layer, cached_inps, cached_oups, config):
    device = next(layer.parameters()).device
    weight_quantizer = layer.weight_fake_quant
    # assert isinstance(weight_quantizer, adaround_quantizer) is True

    weight_quantizer.init(layer.weight.data, config.round_mode)
    w_para = []
    w_opt, w_scheduler = None, None
    w_para += [weight_quantizer.alpha]
    w_opt = torch.optim.Adam(w_para)

    loss_func = LossFunction(layer=layer, weight=config.weight, max_count=config.max_count, b_range=config.b_range,
                             warm_up=config.warm_up)

    world_size = link.get_world_size()
    logger.info('The world size is {}.'.format(world_size))
    '''start training'''
    logger.info('start tuning by adaround')
    sz = len(cached_inps)
    for i in range(config.max_count):
        idx = np.random.randint(0, sz)
        cur_inp = cached_inps[idx]
        cur_inp = [inp.to(device) for inp in cur_inp]
        cur_out = cached_oups[idx].to(device)

        w_opt.zero_grad()
        out_quant = layer(*cur_inp)
        err = loss_func(out_quant, cur_out)
        err /= world_size
        err.backward()
        if world_size > 1:
            for param in w_para:
                link.allreduce(param.grad.data)
        w_opt.step()
        if w_scheduler:
            w_scheduler.step()
    torch.cuda.empty_cache()

    layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
    weight_quantizer.adaround = False


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
    for node in nodes:
        if node.op == "call_module" and isinstance(fp32_modules[node.target], _ADAROUND_SUPPORT_TYPE):
            '''if you want to do layer reconstruction, please do layer_reconstruction'''
            logger.info('prepare layer reconstruction for {}'.format(node.target))
            fp32_module = fp32_modules[node.target]
            quant_module = quant_modules[node.target]
            cached_oups = save_inp_oup_data(fp32_model, fp32_module, cali_data, store_inp=False, store_oup=True,
                                            keep_gpu=config.keep_gpu)
            cached_inps = save_inp_oup_data(quant_model, quant_module, cali_data, store_inp=True, store_oup=False,
                                            keep_gpu=config.keep_gpu)
            layer_reconstruction(quant_module, cached_inps, cached_oups, config)
    return quant_model
