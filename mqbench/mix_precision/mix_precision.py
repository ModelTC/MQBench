from typing import Tuple

from torch.nn import Module

from mqbench.utils.logger import logger
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import disable_all
from mqbench.mix_precision.hessian_per_layer import hessian_per_layer


def mixprecision_profiling(model: Module, quantized_model: Module, data: Tuple, criterion, algo='hawq'):
    """
    Get layer sensitive index.
    A lot of algorithms can do the same thing.
    HAWQ is the most useful one.
    Naive is the most straight forward one.
    """
    if algo is 'hawq':
        sensetive_dict = hawq(model, data, criterion, type='eigenvalues')
    elif algo is 'naive':
        sensetive_dict = prec_degradation_by_layer(model, quantized_model, data, criterion)
    else:
        logger.info("Unknown algorithm!")
    return sensetive_dict


def mp_model_size(model: Module):
    """
    Calcualte model size in different bitwidth.
    """
    mp_size = 0
    for mod in model.modules():
        if hasattr(mod, 'weight_fake_quant'):
            bitwidth = mod.weight_fake_quant.bitwidth
            mp_size += mod.weight.numel() * bitwidth
        elif hasattr(mod, 'weight'):
            mp_size += mod.weight.numel() * 32
    return mp_size / 8 / 1024 / 1024


def prec_degradation_by_layer(model: Module, quantized_model: Module, data: Tuple, creterion):
    """
    Calculate model acc in different bitwidth.
    """
    input_data, label_data = data
    sensetive_dict = {}
    output_data = model(input_data)
    fp_loss = creterion(output_data, label_data)
    model_size = mp_model_size(model)
    logger.info("FP model size: {:.2f} MB".format(model_size))

    for name, mod in quantized_model.named_modules():
        logger.setLevel('CRITICAL')
        disable_all(quantized_model)
        logger.setLevel('INFO')
        if hasattr(mod, 'weight_fake_quant'):
            mod.weight_fake_quant.enable_observer()
            mod.weight_fake_quant.enable_fake_quant()
            output_data = quantized_model(input_data)
            loss = creterion(output_data, label_data)
            sensetive_dict[name] = loss
            logger.info("Layer {} with sensetive {}".format(name, loss - fp_loss))

    return sensetive_dict


def hawq(model: Module, data: Tuple, criterion, type='trace'):
    """
    HAWQ layer sensetive indicator. Using extend PyHessian to calculate.
    """
    inputs, targets = data
    hessian_comp = hessian_per_layer(model, criterion, data=(inputs, targets), cuda=True)
    if type == 'eigenvalues':
        max_eigenvalues_dict = hessian_comp.layer_eigenvalues()
        for name, max_eignevalues in max_eigenvalues_dict.items():
            logger.info("Layer {} with max eigen values: {}".format(name, max_eignevalues))
        return max_eigenvalues_dict
    elif type == 'trace':
        trace_dict = hessian_comp.layer_trace()
        for name, trace in trace_dict.items():
            logger.info("Layer {} with trace: {}".format(name, trace))
        return trace_dict
    else:
        raise(NotImplementedError, "{} is not supported, only trace and eigenvalues.".format(type))


def mixprecision_bit_selection(model: Module, bitwidth_list, sensetive_dict, model_size_constraints, latency_constraints):
    """
    Resolute bitwidth by layer sensetive index / model size / accuracy.
    """
    # preato_frontier(model)
    ILP_bit_selection(bitwidth_list, sensetive_dict, model_size_constraints, latency_constraints)


def ILP_bit_selection(bitwidth_list, sensetive_dict, model_size_constraints, latency_constraints):
    from pulp import LpMinimize
    import pulp
    prob = pulp.LpProblem("Model_Size", LpMinimize)


if __name__ == '__main__':
    import torchvision
    import torch

    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    ce_loss = torch.nn.CrossEntropyLoss()
    inputs = torch.rand(2, 3, 224, 224)
    targets = torch.rand(2, 1000)
    model = model.cuda()
    inputs, targets = inputs.cuda(), targets.cuda()
    quantized_model = prepare_by_platform(model, BackendType.Tensorrt)
    naive_sensetive_dict = mixprecision_profiling(model, quantized_model, data=(inputs, targets), criterion=ce_loss, algo='naive')
    hawq_sensetive_dict = mixprecision_profiling(model, None, data=(inputs, targets), criterion=ce_loss, algo='hawq')

    test_bitwidth_list = [2, 4, 8, 16]