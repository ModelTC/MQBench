from enum import Enum
from typing import Any, Dict

import torch
from torch.fx.symbolic_trace import symbolic_trace
from torch.quantization.quantize_fx import _swap_ff_with_fxff, _fuse_fx
from torch.quantization import QConfig

from mqbench.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize
)
from mqbench.observer import (
    ClipStdObserver,
    LSQObserver,
    MinMaxObserver,
    EMAMinMaxObserver
)
from mqbench.fuser_method_mappings import fuse_custom_config_dict
from mqbench.utils.logger import logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER


class BackendType(Enum):
    Academic = 'Academic'
    Tensorrt = 'Tensorrt'
    SNPE = 'SNPE'
    PPLW8A16 = 'PPLW8A16'
    NNIE = 'NNIE'


class QuantizeScheme(object):
    """Describe quantization scheme.
    """
    def __init__(self, symmetry=True, per_channel=False, pot_scale=False, bit=8):
        self.symmetry = symmetry
        self.per_channel = per_channel
        self.pot_scale = pot_scale
        self.bit = bit
        if self.per_channel:
            self.torch_qscheme = torch.per_channel_symmetric if self.symmetry else torch.per_channel_affine
        else:
            self.torch_qscheme = torch.per_tensor_symmetric if self.symmetry else torch.per_tensor_affine

    def to_observer_params(self):
        return {
            'quant_min': -2 ** (self.bit - 1) if self.symmetry else 0,
            'quant_max': 2 ** (self.bit - 1) - 1 if self.symmetry else 2 ** self.bit - 1,
            'dtype': torch.qint8 if self.symmetry else torch.quint8,
            'pot_scale': self.pot_scale,
            'qscheme': self.torch_qscheme,
            'reduce_range': False,
            'ch_axis': 0 if self.per_channel else -1
        }

    def __str__(self):
        return "Symmetric: {} / Bitwidth: {} / Per channel: {} / Pot scale: {}".format(self.symmetry, 
                                                                                       self.bit,
                                                                                       self.per_channel,
                                                                                       self.pot_scale)


ParamsTable = {
    BackendType.Academic: dict(qtype='affine'),    # noqa: E241
    BackendType.NNIE:     dict(qtype='nnie',       # noqa: E241
                               # NNIE actually do not need w/a qscheme. We add for initialize observer only.
                               w_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                               a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                               default_weight_quantize=NNIEFakeQuantize,
                               default_act_quantize=NNIEFakeQuantize,
                               default_weight_observer=MinMaxObserver,
                               default_act_observer=EMAMinMaxObserver),
    BackendType.Tensorrt: dict(qtype='affine',     # noqa: E241
                               w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                               a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                               default_weight_quantize=LearnableFakeQuantize,
                               default_act_quantize=LearnableFakeQuantize,
                               default_weight_observer=LSQObserver,
                               default_act_observer=LSQObserver),
    BackendType.SNPE:     dict(qtype='affine',     # noqa: E241
                               w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                               a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                               default_weight_quantize=LearnableFakeQuantize,
                               default_act_quantize=LearnableFakeQuantize,
                               default_weight_observer=LSQObserver,
                               default_act_observer=LSQObserver),
    BackendType.PPLW8A16: dict(qtype='affine',     # noqa: E241
                               w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                               a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=16),
                               default_weight_quantize=LearnableFakeQuantize,
                               default_act_quantize=LearnableFakeQuantize,
                               default_weight_observer=LSQObserver,
                               default_act_observer=LSQObserver)
}

ObserverDict = {
    'MinMaxObserver':        MinMaxObserver,                                # noqa: E241
    'EMAMinMaxObserver':     EMAMinMaxObserver,    # More general choice.   # noqa: E241
    'QuantileObserver':      None,                 # TODO quantile.         # noqa: E241
    'ClipStdObserver':       ClipStdObserver,      # Usually used for DSQ.  # noqa: E241
    'LSQObserver':           LSQObserver           # Usually used for LSQ.  # noqa: E241
}

FakeQuantizeDict = {
    'FixedFakeQuantize':     FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241
    'LearnableFakeQuantize': LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    'NNIEFakeQuantize':      NNIEFakeQuantize,       # Quantize function for NNIE   # noqa: E241
    'DoReFaFakeQuantize':    DoReFaFakeQuantize,     # Dorefa                       # noqa: E241
    'DSQFakeQuantize':       DSQFakeQuantize,        # DSQ                          # noqa: E241
    'PACTFakeQuantize':      PACTFakeQuantize        # PACT                         # noqa: E241
}

def get_qconfig_by_platform(deploy_backend: BackendType, extra_qparams):
    """

    Args:
        deploy_backend (BackendType): 
        extra_qparams (dict): 

    >>> extra params format: {
            'w_observer': str, weight observer name,
            'a_observer': str, activation observer name,
            'w_fakequantize': str, weight fake quantize function name,
            'w_fakeq_params": dict, params for weight quantize function,
            'a_fakequantize': str, activation fake quantize function name,
            'a_fakeq_params': dict, params for act quantize function,
            if deploy_backend == BackendType.Academic keys below will be used:
            'w_qscheme': {
                'bit': bitwidth,
                'symmetry': weather quantize scheme is symmetric,
                'per_channel': weather quantize scheme is perchannel,
                'pot_scale': weather scale is power of two.
            }
            'a_qscheme': {
                same with w_qscheme.
            }
        }
    """    
    w_observer = extra_qparams.get('w_observer', None)
    if w_observer:
        w_observer = ObserverDict[w_observer]
    a_observer = extra_qparams.get('a_observer', None)
    if a_observer:
        a_observer = ObserverDict[a_observer]
    w_fakequantize = extra_qparams.get('w_fakequantize', None)
    if w_fakequantize:
        w_fakequantize = FakeQuantizeDict[w_fakequantize]
    a_fakequantize = extra_qparams.get('a_fakequantize', None)
    if w_fakequantize:
        a_fakequantize = FakeQuantizeDict[a_fakequantize]
    backend_params = ParamsTable[deploy_backend]

    # NNIE backend must use NNIEFakeQuantize but leave observer adjustable.
    if backend_params['qtype'] == 'nnie':
        if not w_observer:
            w_observer = backend_params['default_weight_observer']
        if not a_observer:
            a_observer = backend_params['default_act_observer']
        w_qscheme = backend_params['w_qscheme']
        a_qscheme = backend_params['a_qscheme']
        w_config = backend_params['default_weight_quantize'].with_args(observer=w_observer,
                                                                       **w_qscheme.to_observer_params())
        a_config = backend_params['default_act_quantize'].with_args(observer=a_observer,
                                                                    **a_qscheme.to_observer_params())
        return QConfig(activation=a_config, weight=w_config)

    # Academic setting should specific quant scheme in config.
    if deploy_backend == BackendType.Academic:
        w_qscheme = QuantizeScheme(**extra_qparams['w_qscheme'])
        a_qscheme = QuantizeScheme(**extra_qparams['a_qscheme'])
    else:
        w_qscheme = backend_params['w_qscheme']
        a_qscheme = backend_params['a_qscheme']

    # Get weight / act fake quantize function and params.
    if not w_fakequantize:
        w_fakequantize = backend_params['default_weight_quantize']
    w_fakeq_params = extra_qparams.get('w_fakeq_params', {})
    if not a_fakequantize:
        a_fakequantize = backend_params['default_act_quantize']
    a_fakeq_params = extra_qparams.get('a_fakeq_params', {})
    # Observer dot not need extra params for now.
    if not w_observer:
        w_observer = MinMaxObserver
    if not a_observer:
        a_observer = EMAMinMaxObserver

    # Create qconfig.
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    logger.info('Weight Qconfig:\n    FakeQuantize: {} Params: {}\n'
                '    Oberver:      {} Params: {}'.format(w_fakequantize.__name__, w_fakeq_params, 
                                                         w_observer.__name__, str(w_qscheme)))
    logger.info('Activation Qconfig:\n    FakeQuantize: {} Params: {}\n'
                '    Oberver:      {} Params: {}'.format(a_fakequantize.__name__, a_fakeq_params, 
                                                         a_observer.__name__, str(a_qscheme)))
    return QConfig(activation=a_qconfig, weight=w_qconfig)


def prepare_qat_fx_by_platform(
        model: torch.nn.Module,
        deploy_backend: BackendType,
        prepare_custom_config_dict: Dict[str, Any] = {}):
    assert model.training, 'prepare_qat_fx_custom only works for models in  ' + \
        'train mode'

    logger.info("Quantize model using {} scheme.".format(deploy_backend))

    # Get Qconfig
    extra_qconfig_dict = prepare_custom_config_dict.get('extra_qconfig_dict', {})
    qconfig = get_qconfig_by_platform(deploy_backend, extra_qconfig_dict)
    # Preserve attr.
    preserve_attr_dict = dict()
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = model
            if submodule_name != "":
                cur_module = getattr(model, submodule_name)
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            preserve_attr_dict[submodule_name] = {}
            for attr in preserve_attr_list:
                preserve_attr_dict[submodule_name][attr] = getattr(cur_module, attr)
    # Symbolic trace
    concrete_args = prepare_custom_config_dict.get('concrete_args', None)
    graph_module = symbolic_trace(model, concrete_args=concrete_args)
    # Model fusion.
    extra_fuse_dict = prepare_custom_config_dict.get('extra_fuse_dict', {})
    _swap_ff_with_fxff(graph_module)
    extra_fuse_dict.update(fuse_custom_config_dict)
    graph_module = _fuse_fx(graph_module, extra_fuse_dict)
    # Prepare
    import mqbench.custom_quantizer  # noqa: F401
    extra_quantizer_dict = prepare_custom_config_dict.get('extra_quantizer_dict', {})
    quantizer = DEFAULT_MODEL_QUANTIZER[deploy_backend](extra_quantizer_dict)
    prepared = quantizer.prepare(graph_module, qconfig)
    # Restore attr.
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = prepared
            if submodule_name != "":
                cur_module = getattr(prepared, submodule_name)
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            for attr in preserve_attr_list:
                logger.info("Preserve attr: {}.{}".format(submodule_name, attr))
                setattr(cur_module, attr, preserve_attr_dict[submodule_name][attr])
    return prepared               