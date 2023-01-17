import torch

from mqbench.utils.logger import logger


def enable_calibration(model):
    logger.info('Enable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name: 
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.disable_fake_quant()


def enable_calibration_quantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Enable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            if quantizer_type not in name: 
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Enable quant: {}'.format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()


def enable_quantization(model, weight_cali_on=False, act_cali_on=False):
    '''
    We enable all quantization for quantization aware training.
    But we sometimes remain weight calibration on for update minmax all along.
    For some hardware, there is no weight quant param to be set, which mean it will calculate
    min / max for weight.
    Assume weight scale * 127 > abs(weight).max() after some training. Training scale and deploy
    scale can be various, so we have to update range every iter.
    '''
    logger.info('Disable observer and Enable quantize.')
    if weight_cali_on:
        logger.info('Enable observer for weight.')
    if act_cali_on:
        logger.info('Enable observer for activation.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            submodule.enable_fake_quant()
            if weight_cali_on and 'weight_fake_quant' in name:
                logger.debug('Enable observer and Enable quant: {}'.format(name))
                submodule.enable_observer()
            elif act_cali_on and 'act_fake_quant' in name:
                logger.debug('Enable observer and Enable quant: {}'.format(name))
                submodule.enable_observer()
            else:
                logger.debug('Disable observer and Enable quant: {}'.format(name))
                submodule.disable_observer()


def disable_all(model):
    logger.info('Disable observer and Disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Disable observer and Disable quantize: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()


def enable_all(model):
    '''Enable calibration and quantization for every iter, means min / max can be updated
    while training. Use for QAT but can not set range.
    '''
    logger.info('Enable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, torch.quantization.FakeQuantizeBase):
            logger.debug('Enable observer and Enable quantize: {}'.format(name))
            submodule.enable_observer()
            submodule.enable_fake_quant()
