import os
import numpy as np
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.quantization.observer import ObserverBase


from mqbench.utils import deepcopy_graphmodule
from mqbench.utils.state import enable_quantization, disable_all
from mqbench.utils.logger import logger


__all__ = ['adaround']
_ADAROUND_SUPPORT_TYPE = (nn.Conv2d, nn.Linear)


def lp_norm(prediction, target, p=2.0):
    """Function to calculate LP-Norm loss term.

    Args:
        prediction (torch.Tensor): 
        target (torch.Tensor): 
        p (float, optional): Order of norm. Defaults to 2.0.

    Returns:
        torch.Tensor: 
    """
    return (prediction - target).abs().pow(p).sum(1).mean()


def _rectified_sigmoid(x, zeta, gamma):
    """Function to generate rounding mask.

    Args:
        x (torch.Tensor): 
        zeta (torch.Tensor): 
        gamma (torch.Tensor): 

    Returns:
        torch.Tensor: 
    """
    return ((zeta - gamma) * torch.sigmoid(x) + gamma).clamp(0, 1)


def adaround(model: GraphModule, cali_data,
             lr: float = 0.001, batch_size: int = 128, max_iter: int = 8000,
             weight: float = 0.01, beta: float = 20, gamma: float = -0.1, zeta: float = 1.1):
    """Main function to run AdaRound on a given model.

    Args:
        model (GraphModule): Model to adaround.
        cali_data (torch.tensor): Stacked tensor.
        lr (float, optional): Defaults to 0.001.
        batch_size (int, optional): Defaults to 128.
        max_iter (int, optional): Defaults to 8000.
        weight (float, optional): Defaults to 0.01.
        beta (float, optional): Defaults to 20.
        gamma (float, optional): Defaults to -0.1.
        zeta (float, optional): Defaults to 1.1.

    Returns:
        GraphModule: Modified copy of the given model.
    """
    device = cali_data.device
    model.to(device)

    # apply rewritten deepcopy of GraphModule
    quant_model = deepcopy_graphmodule(model)
    quant_model.eval()
    model.eval()

    # insert observer to record input/output
    fp_observer_binding_dict = _insert_observer(model, "output")
    quant_observer_binding_dict = _insert_observer(quant_model, "input")

    logger.info("Record Outputs ...")
    # apply data to record output
    disable_all(model)
    enable_quantization(quant_model)

    saver = FpOutputSaver(model, observer_binding_dict=fp_observer_binding_dict,
                          input_data=cali_data)

    # get layers for reconstruction
    modules = dict(quant_model.named_modules())
    quant_module_name_list = _get_quant_modules_by_topology(quant_model)

    # disable gradient for all parameters
    for p in quant_model.parameters():
        p.requires_grad = False

    # learn the rounding mask for each layer
    for node_name in quant_module_name_list:
        logger.info("Adaround for Layer: {}".format(node_name))
        # get input and output tensors
        output_tensor = saver.get_result_by_name(node_name).to(device)
        input_observer = modules[quant_observer_binding_dict[node_name].name]
        cur_node = _get_node_by_name(quant_model, node_name)
        if cur_node is not None:
            module = modules[cur_node.target]
        else:
            raise RuntimeError("Node not found in graph.")

        with _Recorder(input_observer):
            with torch.no_grad():
                quant_model(cali_data)
            input_tensor = input_observer.cache.detach()

        # optimize the 'alpha'
        temp_anneal = TempDecay(t_max=max_iter, start_b=beta)
        ada_reg_loss = AdaRoundReg(zeta=zeta, gamma=gamma, weight=weight,
                                   temp_anneal=temp_anneal, h_func=_rectified_sigmoid)

        weight_fake_quant = module.weight_fake_quant
        ch_axis = weight_fake_quant.activation_post_process.ch_axis
        scale, zero_point = weight_fake_quant.activation_post_process.calculate_qparams()
        quant_min, quant_max = weight_fake_quant.activation_post_process._calculate_qmin_qmax()
        ada_quantizer = AdaRoundQuantizer(reg=ada_reg_loss, scale=scale, zero_point=zero_point,
                                          quant_min=quant_min, quant_max=quant_max, ch_axis=ch_axis)

        ada_layer = AdaRoundLayer(module, ada_reg_loss, ada_quantizer).to(device)

        alpha = learning_alpha(input_tensor, output_tensor,
                               ada_layer, ada_reg_loss, lr,
                               batch_size, max_iter)

        ada_quantizer.soft_quantize = False
        module.weight.data = ada_quantizer(module.weight, alpha)
        module.weight.requires_grad = False

    _del_tensor_observer(quant_model, quant_observer_binding_dict)

    return quant_model


def _del_tensor_observer(gm: GraphModule, observer_binding_dict):
    modules = dict(gm.named_modules())
    nodes = list(gm.graph.nodes)
    # Quant model tensor observer insert in 'input' mode.
    for node in observer_binding_dict.values():
        delattr(gm, node.name)
        for _node in list(node.users.keys()):
            _node.args = node.args
    for node in observer_binding_dict.values():
        gm.graph.erase_node(node)

    gm.recompile()
    gm.graph.lint()


def _insert_observer(gm: GraphModule, insert_type="input"):
    """Insert observers to record the input and output of target layers.

    Args:
        gm (GraphModule): 
        insert_type (str, optional): Defaults to "input".

    Returns:
        Dict: Dict to lookup the observers.
    """
    assert insert_type in ["input", "output"], "insert_type should be 'input' or 'output'."

    modules = dict(gm.named_modules())
    nodes = list(gm.graph.nodes)

    observer_prefix = "_{}_observer".format(insert_type)
    insert_idx = 0
    observer_binding_dict = {}

    for node in nodes:
        if node.op == "call_module" and isinstance(modules[node.target], _ADAROUND_SUPPORT_TYPE):
            observer_name = observer_prefix + str(insert_idx)

            observer = TensorObserver(recording=(insert_type == "output"))
            setattr(gm, observer_name, observer)
            if insert_type == "input":
                with gm.graph.inserting_before(node):
                    insert_node = gm.graph.create_node("call_module", observer_name, node.args, {})
                node.args = (insert_node, )
            else:
                with gm.graph.inserting_after(node):
                    insert_node = gm.graph.create_node("call_module", observer_name, (node, ), {})
                node.replace_all_uses_with(insert_node)
                insert_node.args = (node, )
            observer_binding_dict[node.name] = insert_node
            insert_idx += 1

    gm.recompile()
    gm.graph.lint()
    return observer_binding_dict


class TensorObserver(ObserverBase):
    recording_enabled: torch.Tensor

    def __init__(self, dtype=torch.float32, recording=True):
        """Observer to record tensors.

        Args:
            dtype (type, optional): Defaults to torch.float32.
            recording (bool, optional): Defaults to True.
        """
        # overwrite 'dtype' attr instead of passing as args directly 
        # to avoid built-in assertion in torch.quantization.observer._ObserverBase
        super(TensorObserver, self).__init__(dtype=torch.quint8)
        self.dtype = dtype
        self._cache = None
        self.register_buffer("recording_enabled", torch.tensor([1], dtype=torch.uint8))
        self.enable_recording(recording)

    def enable_recording(self, enabled=True):
        assert isinstance(enabled, bool)
        self.recording_enabled[0] = 1 if enabled else 0

    def forward(self, x):
        if self.recording_enabled[0] == 1:
            self._cache = x
        return x

    def calculate_qparams(self, **kwargs):
        pass

    @property
    def cache(self):
        return self._cache

    def clear(self):
        self._cache = None


class FpOutputSaver:
    @torch.no_grad()
    def __init__(self, fp_gm: GraphModule,
                 observer_binding_dict: Dict[str, Node],
                 save_loc="disk", root="./cali_data_cache",
                 input_data=None):
        """
        Currently, there are two options provided to save floating point model
        outputs, including saving to disk and caching on GPU memory.

        1) Dump to disk. If so, the output feature maps are dumped to disk
        and loaded to memory when necessary.

        2) Cache on GPU side. If so, the output feature maps are kept in memory.
        This option will be risky with a large network and limited GPU memory.
        """

        assert isinstance(fp_gm, GraphModule), "Input module must be a GraphModule."
        assert save_loc in ["disk", "gpu"], "Saving location should be 'disk' or 'gpu'."

        self.module = fp_gm
        self.observer_binding = observer_binding_dict
        self.save_loc = save_loc
        self.data_root = root
        self._data = dict()

        if self.save_loc == "disk" and not os.path.exists(self.data_root):
            logger.info('Save data on disk, create directory {}'.format(self.data_root))
            os.mkdir(self.data_root)
        saving_operation = self._disk_saving_operation \
            if self.save_loc == "disk" else self._gpu_saving_operation

        self._save(input_data=input_data, saving_operation=saving_operation)

    @torch.no_grad()
    def _save(self, input_data, saving_operation: Callable):
        modules = dict(self.module.named_modules())
        self._turn_on_all_observers(True)
        self.module(input_data)

        for node_name, observer_node in self.observer_binding.items():
            observer = modules[observer_node.target]
            observer.enable_recording(False)
            # Do saving operation and clear cache.
            saving_operation(observer, node_name)
            observer.clear()

        self._turn_on_all_observers(False)

    def _disk_saving_operation(self, observer: TensorObserver, node_name: str):
        output_numpy = observer.cache.cpu().numpy()
        np.save(os.path.join(self.data_root, "{}.npy".format(node_name)),
                output_numpy)

    def _gpu_saving_operation(self, observer: TensorObserver, node_name: str):
        self._data[node_name] = observer.cache

    def _turn_on_all_observers(self, recording=True):
        modules = dict(self.module.named_modules())
        for node in self.module.graph.nodes:
            if node.op == "call_module":
                if isinstance(modules[node.target], TensorObserver):
                    modules[node.target].enable_recording(recording)

    def get_result_by_name(self, node_name):
        assert node_name in self.observer_binding
        if self.save_loc == "disk":
            # Load from file according to node_name.
            output_numpy = np.load(os.path.join(self.data_root,
                                   "{}.npy".format(node_name)))
            saved_data = torch.from_numpy(output_numpy)
            saved_data = saved_data
            return saved_data

        elif self.save_loc == "gpu":
            # Load from GPU memory.
            return self._data[node_name]


def _get_quant_modules_by_topology(gm: GraphModule):
    """Get modules in the model.

    Args:
        gm (GraphModule): 

    Returns:
        list: 
    """
    module_name_list = []
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            if isinstance(modules[node.target], _ADAROUND_SUPPORT_TYPE):
                module_name_list.append(node.name)
    return module_name_list


def _get_node_by_name(gm: GraphModule, node_name: str):
    """

    Args:
        gm (GraphModule): 
        node_name (str): 

    Returns:
        torch.fx.Node: 
    """
    for node in gm.graph.nodes:
        if node.name == node_name:
            return node
    return None


class _Recorder:
    def __init__(self, observer: TensorObserver):
        self.observer = observer
        self.observer.enable_recording(True)
        self.observer.clear()

    def __enter__(self):
        pass

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.observer.enable_recording(False)
        self.observer.clear()


class AdaRoundReg():
    def __init__(self, zeta=1.1, gamma=-0.1, weight=0.01,
                 temp_anneal: Callable = None, h_func: Callable = _rectified_sigmoid):
        super(AdaRoundReg, self).__init__()
        self.zeta = zeta
        self.gamma = gamma
        self.weight = weight
        self.temp_anneal = temp_anneal
        self.h_func = h_func
        self.beta = None

    def round_mask(self, alpha):
        return self.h_func(alpha, self.zeta, self.gamma)

    def loss(self, alpha, iter_num):
        self.beta = self.temp_anneal(iter_num)
        return self.weight * (1 - torch.pow((self.round_mask(alpha)
                                            - 0.5).abs() * 2, self.beta)).sum()


class TempDecay:
    def __init__(self, t_max=10000, rel_start_decay=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + 0.5 * (self.start_b 
                                       - self.end_b) * (1 + np.cos(rel_t * np.pi))


class AdaRoundQuantizer:
    def __init__(self, reg: AdaRoundReg, scale, zero_point,
                 quant_min=-128, quant_max=127, ch_axis=-1,
                 soft=True):
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.ch_axis = ch_axis

        self.zeta = reg.zeta
        self.gamma = reg.gamma

        self.soft_quantize = soft
        self.scale = scale
        self.zero_point = zero_point

        self.h_func = reg.h_func

    def __call__(self, w, alpha):
        scale = self.scale
        zero_point = self.zero_point

        if self.soft_quantize:
            w = (w / scale).floor() + self.h_func(alpha, self.zeta, self.gamma)
        else:
            w = (w / scale).floor() + (alpha > 0).float()

        w += zero_point
        w = w.clamp(self.quant_min, self.quant_max)
        w -= zero_point

        w = w * scale
        return w


class AdaRoundLayer(nn.Module):
    def __init__(self, module: nn.Module,
                 reg: AdaRoundReg, quantizer: AdaRoundQuantizer):
        super(AdaRoundLayer, self).__init__()
        assert isinstance(module, _ADAROUND_SUPPORT_TYPE), \
            "Cannot apply AdaRound on this module."

        self.module = module
        self.quantizer = quantizer

        if self.module.bias is not None:
            self.module.bias.requires_grad = False

        if self.quantizer.ch_axis != -1:
            new_shape = [1] * len(self.module.weight.shape)
            new_shape[self.quantizer.ch_axis] = self.module.weight.shape[self.quantizer.ch_axis]
            self.quantizer.scale = self.quantizer.scale.reshape(new_shape)
            self.quantizer.zero_point = self.quantizer.zero_point.reshape(new_shape)

        # Init rest.
        scale = self.quantizer.scale
        rest = self.module.weight / scale - (self.module.weight / scale).floor()
        rest = -torch.log((reg.zeta - reg.gamma) / (rest - reg.gamma) - 1)
        self.alpha = torch.nn.Parameter(rest, True)

    def forward(self, x):
        weight = self.quantizer(self.module.weight, self.alpha)

        if isinstance(self.module, nn.Conv2d):
            x = F.conv2d(x, weight, self.module.bias, stride=self.module.stride,
                         padding=self.module.padding, dilation=self.module.dilation,
                         groups=self.module.groups)
        elif isinstance(self.module, nn.Linear):
            x = F.linear(x, weight, self.module.bias)
        else:
            raise RuntimeError("Unsupported module type.")

        if isinstance(self.module, (torch.nn.intrinsic.qat.ConvReLU2d, 
                                    torch.nn.intrinsic.qat.LinearReLU)):
            x = F.relu(x)

        return x  


# TODO: support different loss functions / p in lp_norm
def learning_alpha(in_tensor: torch.Tensor,
                   fp_out_tensor: torch.Tensor,
                   ada_layer: AdaRoundLayer,
                   ada_reg: AdaRoundReg,
                   learning_rate: float,
                   batch_size: int,
                   max_iter: int) -> torch.Tensor:

    optimizer = torch.optim.Adam([ada_layer.alpha])

    for epoch in range(max_iter):
        for idx in range(np.ceil(len(in_tensor) / batch_size).astype(int)):
            st = idx * batch_size
            ed = st + batch_size

            input_ = in_tensor[st:ed].squeeze(1).detach()
            fp_output = fp_out_tensor[st:ed].squeeze(1).detach()
            output = ada_layer(input_) 

            loss_p = lp_norm(output, fp_output)
            loss_reg = ada_reg.loss(ada_layer.alpha, epoch)
            loss = loss_p + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            logger.info("Epoch: {:<4} L2 Loss: {:>10.3f} Loss P: "
                        "{:>8.6f} Loss Reg: {:>5.3f} Beta: {:>3.3f}".format(epoch, loss, loss_p, 
                                                                            loss_reg, ada_reg.beta))
    res = ada_reg.round_mask(ada_layer.alpha)
    logger.info("Loss: {:>5.3f} Ceil: {:>5} Floor: {:>5} Total: {:>5} Ratio: {:>.3f}".format(
        loss,
        res[res + 1e-4 >= 1.0].numel(), res[res <= 1e-4].numel(), torch.numel(res),
        (res[res + 1e-4 >= 1.0].numel() + res[res <= 1e-4].numel()) / torch.numel(res)))
    return ada_layer.alpha