from typing import Tuple

from torch.nn import Module
import numpy as np

from mqbench.utils.logger import logger
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import disable_all
from mqbench.mix_precision.hessian_per_layer import hessian_per_layer


def mixprecision_profiling(model: Module, quantized_model: Module, data: Tuple, criterion, algo='naive'):
    """
    Get layer sensitive index.
    A lot of algorithms can do the same thing.
    HAWQ is the most useful one.
    Naive is the most straight forward one.
    """
    layer_parameters_dict = {}
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear)):
            layer_parameters_dict[name] = mod.weight.numel()
    model_size = sum(list(layer_parameters_dict.values())) * 32 / 8 / 1024 / 1024
    logger.info("FP model size: {:.2f} MB".format(model_size))

    if algo is 'hawq_eigen':
        sensetive_dict = hawq(model, data, criterion, type='eigenvalues')
        # Do normalize.
        for layer, index in sensetive_dict.items():
            sensetive_dict[layer] = index / layer_parameters_dict[layer]
        for name, max_eignevalues in sensetive_dict.items():
            logger.info("Layer {} with max eigen values: {}".format(name, max_eignevalues))
    elif algo is 'hawq_trace':
        sensetive_dict = hawq(model, data, criterion, type='trace')
        # Do normalize.
        for layer, index in sensetive_dict.items():
            sensetive_dict[layer] = index / layer_parameters_dict[layer]
        for name, trace in sensetive_dict.items():
            logger.info("Layer {} with trace: {}".format(name, trace))
    elif algo is 'naive':
        sensetive_dict = prec_degradation_by_layer(model, quantized_model, data, criterion)
    else:
        logger.info("Unknown algorithm!")
    return sensetive_dict, layer_parameters_dict


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
        return max_eigenvalues_dict
    elif type == 'trace':
        trace_dict = hessian_comp.layer_trace()
        return trace_dict
    else:
        raise(NotImplementedError, "{} is not supported, only trace and eigenvalues.".format(type))


def mixprecision_bit_selection(bitwidth_list, sensetive_dict, layer_parameters_dict, delta_w, model_size_constraints, latency_constraints):
    """
    Resolute bitwidth by layer sensetive index / model size / accuracy.
    """
    # preato_frontier(model)
    ILP_bit_selection(bitwidth_list, sensetive_dict, layer_parameters_dict, delta_w, model_size_constraints, latency_constraints)


def ILP_bit_selection(bitwidth_list, sensetive_dict, layer_parameters_dict, delta_w, model_size_constraints: int, latency_constraints: int):
    """
    Bit selection process using ILP.
    """
    from pulp import LpMinimize, LpVariable, LpInteger, LpStatus, GLPK_CMD, value
    import pulp

    assert model_size_constraints or latency_constraints

    prob = pulp.LpProblem("Min model size with best acc", LpMinimize)
    variable = {}
    for layer_name in sensetive_dict:
        for bit in bitwidth_list:
            variable[f"x_{layer_name}_{bit}"] = LpVariable(f"x_{layer_name}_{bit}", 0, 1, cat=LpInteger)

    # Model acc constrains
    senseitve_contrains = []
    for name, params in layer_parameters_dict.items():
        for bit in bitwidth_list:
            senseitve_contrains.append(variable[f"x_{name}_{bit}"] * delta_w[name][bit])
    prob += sum(senseitve_contrains)

    # Every Layer can only be assigned to one bitwidth.
    for layer_name in sensetive_dict:
        prob += sum([variable[f"x_{layer_name}_{bit}"] for bit in bitwidth_list]) == 1

    # Model size constrains
    total_size = []
    for name, params in layer_parameters_dict.items():
        for bit in bitwidth_list:
            total_size.append(variable[f"x_{name}_{bit}"] * bit * params)
    prob += sum(total_size) <= model_size_constraints * 8 * 1024 * 1024

    status = prob.solve(GLPK_CMD(msg=1, options=["--tmlim", "10000","--simplex"]))
    LpStatus[status]
    for layer_name in sensetive_dict:
        resigned = False
        for bit in bitwidth_list:
            if value(variable[f"x_{layer_name}_{bit}"]) == 1:
                if resigned:
                    logger.warning("Bad")
                logger.info("Layer {} with {} bits".format(layer_name, bit))
                resigned = True
    total_size = []
    for name, params in layer_parameters_dict.items():
        for bit in bitwidth_list:
            total_size.append(value(variable[f"x_{name}_{bit}"]) * bit * params)
    logger.info("Result model size {} MB.".format(sum(total_size) / 8 / 1024 / 1024))

    senseitve_contrains = []
    for name, params in layer_parameters_dict.items():
        for bit in bitwidth_list:
            senseitve_contrains.append(value(variable[f"x_{name}_{bit}"]) * delta_w[name][bit])
    logger.info("Result model sensetive is {}".format(sum(senseitve_contrains)))


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
    naive_sensetive_dict, layer_parameters_dict = mixprecision_profiling(model, quantized_model, data=(inputs, targets), criterion=ce_loss, algo='naive')
    hawq_sensetive_dict, _ = mixprecision_profiling(model, None, data=(inputs, targets), criterion=ce_loss, algo='hawq_eigen')
    hawq_sensetive_dict, _ = mixprecision_profiling(model, None, data=(inputs, targets), criterion=ce_loss, algo='hawq_trace')

    test_bitwidth_list = [1, 2, 4, 8, 16, 32]
    test_delta_w = {}
    for k, v in hawq_sensetive_dict.items():
        test_delta_w[k] = {}
        for bit in test_bitwidth_list:
            test_delta_w[k][bit] = abs(v.item()) / (bit / 32)
    mixprecision_bit_selection(test_bitwidth_list,
                               hawq_sensetive_dict,
                               layer_parameters_dict,
                               test_delta_w,
                               model_size_constraints=2,
                               latency_constraints=None)