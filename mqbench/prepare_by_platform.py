from copy import deepcopy
from enum import Enum
from typing import Any, Dict
import types
import inspect

import torch
from torch.fx import Tracer
from torch.fx.graph_module import GraphModule
from torch.quantization.quantize_fx import _swap_ff_with_fxff
from torch.quantization import QConfig
import torch.nn.intrinsic as nni
import mqbench.nn.intrinsic as qnni

from mqbench.fake_quantize import (
    LearnableFakeQuantize,
    NNIEFakeQuantize,
    FixedFakeQuantize,
    DoReFaFakeQuantize,
    DSQFakeQuantize,
    PACTFakeQuantize,
    TqtFakeQuantize,
    AdaRoundFakeQuantize,
    QDropFakeQuantize,
    E4M3FakeQuantize,
    E5M2FakeQuantize,
    GPTQFakeQuantize,
    FP4FakeQuantize,
    GPTQFP4FakeQuantize
)
from mqbench.observer import (
    ClipStdObserver,
    LSQObserver,
    MinMaxFloorObserver,
    MinMaxObserver,
    EMAMinMaxObserver,
    PoTModeObserver,
    EMAQuantileObserver,
    MSEObserver,
    EMAMSEObserver,
    KLDObserver,
)
import mqbench
from mqbench.fuser_method_mappings import fuse_custom_config_dict
from mqbench.utils.logger import logger
from mqbench.utils.registry import DEFAULT_MODEL_QUANTIZER
from mqbench.scheme import QuantizeScheme
import mqbench.nn.intrinsic.qat as qnniqat

__all__ = ['prepare_by_platform']

class BackendType(Enum):
    Academic = 'Academic'
    Tensorrt = 'Tensorrt'
    SNPE = 'SNPE'
    PPLW8A16 = 'PPLW8A16'
    NNIE = 'NNIE'
    Vitis = 'Vitis'
    ONNX_QNN = 'ONNX_QNN'
    PPLCUDA = 'PPLCUDA'
    OPENVINO = 'OPENVINO'
    Tensorrt_NLP = "Tensorrt_NLP"
    Academic_NLP = "Academic_NLP"
    Sophgo_TPU = "Sophgo_TPU"


ParamsTable = {
    BackendType.Academic:   dict(qtype='affine',
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=E4M3FakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),    # noqa: E241
    BackendType.NNIE:       dict(qtype='nnie',       # noqa: E241
                                 # NNIE actually do not need w/a qscheme. We add for initialize observer only.
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=NNIEFakeQuantize,
                                 default_act_quantize=NNIEFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.Tensorrt:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8, symmetric_range=True),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8, symmetric_range=True),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.OPENVINO:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.SNPE:       dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.PPLW8A16:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=16),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver),
    BackendType.Vitis:      dict(qtype='vitis',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=True, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=True, bit=8),
                                 default_weight_quantize=TqtFakeQuantize,
                                 default_act_quantize=TqtFakeQuantize,
                                 default_weight_observer=MinMaxFloorObserver,
                                 default_act_observer=PoTModeObserver),
    BackendType.ONNX_QNN:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=MinMaxObserver),
    BackendType.PPLCUDA:    dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=False, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=False, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=MinMaxObserver),
    BackendType.Sophgo_TPU:   dict(qtype='affine',     # noqa: E241
                                 w_qscheme=QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8),
                                 a_qscheme=QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8),
                                 default_weight_quantize=LearnableFakeQuantize,
                                 default_act_quantize=LearnableFakeQuantize,
                                 default_weight_observer=MinMaxObserver,
                                 default_act_observer=EMAMinMaxObserver)
}
ParamsTable[BackendType.Tensorrt_NLP] = ParamsTable[BackendType.Tensorrt]
ParamsTable[BackendType.Academic_NLP] = ParamsTable[BackendType.Academic]

ObserverDict = {
    'MinMaxObserver':           MinMaxObserver,                                    # noqa: E241
    'EMAMinMaxObserver':        EMAMinMaxObserver,        # More general choice.   # noqa: E241
    'MinMaxFloorObserver':      MinMaxFloorObserver,      # For Vitis HW           # noqa: E241
    'PoTModeObserver':          PoTModeObserver,   # For Vitis HW           # noqa: E241
    'EMAQuantileObserver':      EMAQuantileObserver,      # Quantile observer.     # noqa: E241
    'ClipStdObserver':          ClipStdObserver,          # Usually used for DSQ.  # noqa: E241
    'LSQObserver':              LSQObserver,              # Usually used for LSQ.  # noqa: E241
    'MSEObserver':              MSEObserver,                                       # noqa: E241
    'EMAMSEObserver':           EMAMSEObserver,                                    # noqa: E241
    'KLDObserver':              KLDObserver,
}

FakeQuantizeDict = {
    'FixedFakeQuantize': FixedFakeQuantize,      # Unlearnable scale/zeropoint  # noqa: E241
    'LearnableFakeQuantize': LearnableFakeQuantize,  # Learnable scale/zeropoint    # noqa: E241
    'NNIEFakeQuantize':      NNIEFakeQuantize,       # Quantize function for NNIE   # noqa: E241
    'DoReFaFakeQuantize':    DoReFaFakeQuantize,     # Dorefa                       # noqa: E241
    'DSQFakeQuantize':       DSQFakeQuantize,        # DSQ                          # noqa: E241
    'PACTFakeQuantize':      PACTFakeQuantize,       # PACT                         # noqa: E241
    'TqtFakeQuantize':       TqtFakeQuantize,        # TQT                          # noqa: E241
    'AdaRoundFakeQuantize':  AdaRoundFakeQuantize,   # AdaRound                     # noqa: E241
    'QDropFakeQuantize':     QDropFakeQuantize,      # BRECQ & QDrop                # noqa: E241
    'E4M3FakeQuantize':      E4M3FakeQuantize,
    'E5M2FakeQuantize':      E5M2FakeQuantize,
    'GPTQFakeQuantize':      GPTQFakeQuantize, 
    'FP4FakeQuantize':       FP4FakeQuantize,
    'GPTQFP4FakeQuantize':   GPTQFP4FakeQuantize  
}


def get_qconfig_by_platform(deploy_backend: BackendType, extra_qparams: Dict, work_mode: str):
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
                'symmetry': whether quantize scheme is symmetric,
                'per_channel': whether quantize scheme is perchannel,
                'pot_scale': whether scale is power of two.
            }
            'a_qscheme': {
                same with w_qscheme.
            }
            "object_type": [
                (torch.add, qconfig)
            ],
            "module_name": [
                ("conv1", qconfig)
            ]
        }
    """
    w_observer = extra_qparams.get('w_observer', None)
    if w_observer:
        assert w_observer in ObserverDict, \
            'Do not support observer name: {}'.format(w_observer)
        w_observer = ObserverDict[w_observer]
    a_observer = extra_qparams.get('a_observer', None)
    if a_observer:
        assert a_observer in ObserverDict, \
            'Do not support observer name: {}'.format(a_observer)
        a_observer = ObserverDict[a_observer]
    w_fakequantize = extra_qparams.get('w_fakequantize', None)
    if w_fakequantize:
        assert w_fakequantize in FakeQuantizeDict, \
            'Do not support fakequantize name: {}'.format(w_fakequantize)
        w_fakequantize = FakeQuantizeDict[w_fakequantize]
    a_fakequantize = extra_qparams.get('a_fakequantize', None)
    if a_fakequantize:
        assert a_fakequantize in FakeQuantizeDict, \
            'Do not support fakequantize name: {}'.format(a_fakequantize)
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
        qconfig = {'': QConfig(activation=a_config, weight=w_config)}
        object_type = extra_qparams.get('object_type', None)
        if object_type is not None:
            qconfig["object_type"] = object_type
        module_name = extra_qparams.get('module_name', None)
        if module_name is not None:
            qconfig["module_name"] = module_name
        return qconfig

    # Academic setting should specific quant scheme in config.
    if deploy_backend in [BackendType.Academic, BackendType.Academic_NLP]:
        w_qscheme = QuantizeScheme(**extra_qparams['w_qscheme'])
        a_qscheme = QuantizeScheme(**extra_qparams['a_qscheme'])
    else:
        w_qscheme = extra_qparams.get('w_qscheme', None)
        if w_qscheme is None:
            w_qscheme = backend_params['w_qscheme']
        else:
            logger.info("Weight Quant Scheme is overrided!")
            w_qscheme = QuantizeScheme(**w_qscheme)
        a_qscheme = extra_qparams.get('a_qscheme', None)
        if a_qscheme is None:
            a_qscheme = backend_params['a_qscheme']
        else:
            logger.info("Activation Quant Scheme is overrided!")
            a_qscheme = QuantizeScheme(**a_qscheme)

    # Set extra args for observers.
    w_observer_extra_args = extra_qparams.get('w_observer_extra_args', {})
    a_observer_extra_args = extra_qparams.get('a_observer_extra_args', {})
    w_qscheme.kwargs.update(w_observer_extra_args)
    a_qscheme.kwargs.update(a_observer_extra_args)
    # Get weight / act fake quantize function and params. And bias fake quantizer if needed(Vitis)
    if not w_fakequantize:
        w_fakequantize = backend_params['default_weight_quantize']
    w_fakeq_params = extra_qparams.get('w_fakeq_params', {})
    if not a_fakequantize:
        a_fakequantize = backend_params['default_act_quantize']
    a_fakeq_params = extra_qparams.get('a_fakeq_params', {})
    # Get default observer type.
    if not w_observer:
        w_observer = backend_params['default_weight_observer']
    if not a_observer:
        a_observer = backend_params['default_act_observer']

    # Create qconfig.
    # here, rewrited by with_args
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    logger.info('Weight Qconfig:\n    FakeQuantize: {} Params: {}\n'
                '    Oberver:      {} Params: {}'.format(w_fakequantize.__name__, w_fakeq_params,
                                                         w_observer.__name__, str(w_qscheme)))
    logger.info('Activation Qconfig:\n    FakeQuantize: {} Params: {}\n'
                '    Oberver:      {} Params: {}'.format(a_fakequantize.__name__, a_fakeq_params,
                                                         a_observer.__name__, str(a_qscheme)))
    if backend_params['qtype'] == 'vitis':
        logger.info('Bias Qconfig:\n    TqtFakeQuantize with MinMaxObserver')
    
    if deploy_backend in [BackendType.Academic, BackendType.Academic_NLP]:
       return {'':QConfig(activation=a_qconfig, weight=w_qconfig)}
    
    qconfig = {'': QConfig(activation=a_qconfig, weight=w_qconfig)}    
    if deploy_backend == BackendType.Sophgo_TPU:
        qconfig["object_type"] = {torch.nn.Linear:createQConfigForSophgoLiner()} #int8 qat, Sophgo_TPU use sym per-layer
        if work_mode == 'all_int4_qat':
            qconfig["object_type"][torch.nn.Linear] = createQConfigForSophgoLiner(bit_num = 4)
        if work_mode in ['int4_and_int8_mix', 'int4_and_int8_mix_no_fc']:
            w_qscheme = {
                'bit': 4,
                'symmetry': True,
                'per_channel': True,
                'pot_scale': False
            }
            a_qscheme = {
                'bit': 4,
                'symmetry': True,
                'per_channel': False,
                'pot_scale': False
            }
            int4_qconfig = createQConfig(w_qscheme = w_qscheme, a_qscheme = a_qscheme)
            qconfig["object_type"][torch.nn.Conv2d] = int4_qconfig
            from mqbench.custom_quantizer.sophgo_tpu_quantizer import SophgoTpuQuantizer

            import torch.nn as nn
            additional_qat_module_mapping = [
                nni.ConvBn2d,
                nni.ConvBnReLU2d,
                nn.Conv2d,
                nni.ConvReLU2d,
                qnni.ConvTransposeBnReLU2d,
                qnni.ConvTransposeReLU2d,
                qnni.ConvTransposeBn2d
            ]
            for i in additional_qat_module_mapping:
                qconfig["object_type"][i] = int4_qconfig
            if work_mode == 'int4_and_int8_mix_no_fc':
                for i in SophgoTpuQuantizer({}, {})._layers_need_check_is_dw:
                    qconfig["object_type"][i] = int4_qconfig
            else:
                for i in SophgoTpuQuantizer({}, {})._layers_need_scale_form_input_fake_quantizer:
                    qconfig["object_type"][i] = int4_qconfig
            if work_mode != 'int4_and_int8_mix_no_fc':
                liner_qconfig = createQConfigForInt4SophgoLiner()
                qconfig["object_type"][nni.LinearReLU] = liner_qconfig
                qconfig["object_type"][qnni.LinearBn1d] = liner_qconfig
                qconfig["object_type"][torch.nn.Linear] = liner_qconfig
    object_type = extra_qparams.get('object_type', None)
    if object_type is not None:
        if "object_type" in qconfig:
            qconfig["object_type"].update(object_type)
        else:
            qconfig["object_type"] = object_type
            
    module_name = extra_qparams.get('module_name', None)
    if module_name is not None:
        qconfig["module_name"] = module_name
    return qconfig

#LearnableFakeQuantize, LearnableFakeQuantize, MinMaxObserver, EMAMinMaxObserver
# 'w_qscheme': {
#     'bit': bitwidth,
#     'symmetry': whether quantize scheme is symmetric,
#     'per_channel': whether quantize scheme is perchannel,
#     'pot_scale': whether scale is power of two.
# }
# 'a_qscheme': {
#     'bit': bitwidth,
#     'symmetry': whether quantize scheme is symmetric,
#     'per_channel': whether quantize scheme is perchannel,
#     'pot_scale': whether scale is power of two.
# }

def createQConfigForSophgoLiner(bit_num = 8, w_fakequantize = 'LearnableFakeQuantize', w_observer = 'MinMaxObserver', w_fakeq_params = {}, w_observer_extra_args = {}):
    w_observer = ObserverDict[w_observer]
    w_fakequantize = FakeQuantizeDict[w_fakequantize]
    w_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=bit_num) #Sophgo_TPU use sym per-layer
    w_qscheme.kwargs.update(w_observer_extra_args)
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())
    return QConfig(activation=torch.nn.Identity, weight=w_qconfig) #activation use global quant conifg

def createQConfig(w_fakequantize = 'LearnableFakeQuantize', a_fakequantize = 'LearnableFakeQuantize', 
                w_observer = 'MinMaxObserver', a_observer = 'EMAMinMaxObserver', w_qscheme = {}, a_qscheme = {},
                w_fakeq_params = {}, a_fakeq_params = {}, w_observer_extra_args = {}, a_observer_extra_args = {}):
    w_observer = ObserverDict[w_observer]
    w_fakequantize = FakeQuantizeDict[w_fakequantize]
    if w_qscheme is not None:
        w_qscheme = QuantizeScheme(**w_qscheme)
    else:
        w_qscheme = QuantizeScheme(symmetry=True, per_channel=True, pot_scale=False, bit=8)

    w_qscheme.kwargs.update(w_observer_extra_args)
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())

    a_observer = ObserverDict[a_observer]
    a_fakequantize = FakeQuantizeDict[a_fakequantize]
    if a_qscheme is not None:
        a_qscheme = QuantizeScheme(**a_qscheme)
    else:
        a_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=8)
    a_qscheme.kwargs.update(a_observer_extra_args)
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    return QConfig(activation=a_qconfig, weight=w_qconfig)

def createQConfigForInt4SophgoLiner(w_fakequantize = 'LearnableFakeQuantize', a_fakequantize = 'LearnableFakeQuantize', 
                w_observer = 'MinMaxObserver', a_observer = 'EMAMinMaxObserver', w_qscheme = {}, a_qscheme = {},
                w_fakeq_params = {}, a_fakeq_params = {}, w_observer_extra_args = {}, a_observer_extra_args = {}):
    w_observer = ObserverDict[w_observer]
    w_fakequantize = FakeQuantizeDict[w_fakequantize]
    w_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=4)

    w_qscheme.kwargs.update(w_observer_extra_args)
    w_qconfig = w_fakequantize.with_args(observer=w_observer, **w_fakeq_params, **w_qscheme.to_observer_params())

    a_observer = ObserverDict[a_observer]
    a_fakequantize = FakeQuantizeDict[a_fakequantize]
    a_qscheme = QuantizeScheme(symmetry=True, per_channel=False, pot_scale=False, bit=4)
    a_qscheme.kwargs.update(a_observer_extra_args)
    a_qconfig = a_fakequantize.with_args(observer=a_observer, **a_fakeq_params, **a_qscheme.to_observer_params())
    return QConfig(activation=a_qconfig, weight=w_qconfig)

class CustomedTracer(Tracer):
    """
    ``Tracer`` is the class that implements the symbolic tracing functionality
    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
    to ``Tracer().trace(m)``.
    This Tracer override the ``is_leaf_module`` function to make symbolic trace
    right in some cases.
    """
    def __init__(self, *args, customed_leaf_module=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.customed_leaf_module = customed_leaf_module

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        """
        A method to specify whether a given ``nn.Module`` is a "leaf" module.
        Leaf modules are the atomic units that appear in
        the IR, referenced by ``call_module`` calls. By default,
        Modules in the PyTorch standard library namespace (torch.nn)
        are leaf modules. All other modules are traced through and
        their constituent ops are recorded, unless specified otherwise
        via this parameter.
        Args:
            m (Module): The module being queried about
            module_qualified_name (str): The path to root of this module. For example,
                if you have a module hierarchy where submodule ``foo`` contains
                submodule ``bar``, which contains submodule ``baz``, that module will
                appear with the qualified name ``foo.bar.baz`` here.
        """
        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
            return True
        return m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)

def duplicate_reused_nodes(graph: torch.fx.Graph, modules: Dict[str, Any] = {}):
    _dup_prefix = '_dup'
    target_dict = dict()
    dup_modules = dict()
    for node in graph.nodes:
        if node.op == "call_module":
            if node.target not in target_dict:
                target_dict[node.target] = [node]
            else:
                target_dict[node.target].append(node)
    for key in target_dict:
        if len(target_dict[key]) > 1:
            for idx, node in enumerate(target_dict[key]):
                if idx == 0:
                    continue
                module = deepcopy(modules[node.target])
                node.target += _dup_prefix + str(idx)
                dup_modules[node.target] = module
    graph.lint()
    return graph, dup_modules

def prepare_constant_dict(graph: torch.fx.Graph, model: torch.nn.Module):
    def _get_attrs(target, attrs):
        attrs = attrs.split('.')
        for att in attrs:
            target = getattr(target, att)
        return target
    constant_dict = dict()
    for node in graph.nodes:
        if node.op == 'get_attr':
            constant_dict[node.target] = _get_attrs(model, node.target)
    return constant_dict


def prepare_by_platform(
        model: torch.nn.Module,
        deploy_backend: BackendType,
        input_shape_dict: list = None,
        prepare_custom_config_dict: Dict[str, Any] = {},
        custom_tracer: Tracer = None):
    """
    Args:
        model (torch.nn.Module):
        deploy_backend (BackendType):

    >>> prepare_custom_config_dict : {
            extra_qconfig_dict : Dict, Find explanations in get_qconfig_by_platform,
            extra_quantizer_dict: Extra params for quantizer.
            preserve_attr: Dict, Specify attribute of model which should be preserved
                after prepare. Since symbolic_trace only store attributes which is
                in forward. If model.func1 and model.backbone.func2 should be preserved,
                {"": ["func1"], "backbone": ["func2"] } should work.
            Attr below is inherited from Pytorch.
            concrete_args: Specify input for model tracing.
            extra_fuse_dict: Specify extra fusing patterns and functions.
        }

    """
    model_mode = 'Training' if model.training else 'Eval'
    logger.info("Quantize model Scheme: {} Mode: {}".format(deploy_backend, model_mode))

    # Get Qconfig
    extra_qconfig_dict = prepare_custom_config_dict.get('extra_qconfig_dict', {})
    work_mode = prepare_custom_config_dict.get('work_mode', '')
    qconfig = get_qconfig_by_platform(deploy_backend, extra_qconfig_dict, work_mode)

    _swap_ff_with_fxff(model)
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
    customed_leaf_module = prepare_custom_config_dict.get('leaf_module', [])
    tracer = CustomedTracer(customed_leaf_module=tuple(customed_leaf_module))
    if custom_tracer is not None:
        tracer = custom_tracer
    graph = tracer.trace(model, concrete_args)
    print('>>>>>trace graph:',graph)
    name = model.__class__.__name__ if isinstance(model, torch.nn.Module) else model.__name__
    modules = dict(model.named_modules())
    print('>>>>>named_modules:',modules[''])
    graph, duplicated_modules = duplicate_reused_nodes(graph, modules)
    constant_nodes = prepare_constant_dict(graph, model)
    modules.update(duplicated_modules)
    modules.update(constant_nodes)
    graph_module = GraphModule(modules, graph, name)
    if input_shape_dict is not None:
        try:
            from torch.fx.passes import shape_prop
            dev = next(model.parameters()).device
            dummy_input = [torch.rand(shape).to(dev) for shape in input_shape_dict]
            shape_prop.ShapeProp(graph_module).propagate(*dummy_input)
        except:
            print('waring, shape_prop fail')
    # Model fusion.
    extra_fuse_dict = prepare_custom_config_dict.get('extra_fuse_dict', {})
    extra_fuse_dict.update(fuse_custom_config_dict)
    # Prepare
    import mqbench.custom_quantizer  # noqa: F401
    extra_quantizer_dict = prepare_custom_config_dict.get('extra_quantizer_dict', {})
    quantizer = DEFAULT_MODEL_QUANTIZER[deploy_backend](extra_quantizer_dict, extra_fuse_dict)
    prepared = quantizer.prepare(graph_module, qconfig)
    # Restore attr.
    if 'preserve_attr' in prepare_custom_config_dict:
        for submodule_name in prepare_custom_config_dict['preserve_attr']:
            cur_module = prepared
            _type = type(model)
            if submodule_name != "":
                cur_module = getattr(prepared, submodule_name)
                _type = type(getattr(model, submodule_name))
            preserve_attr_list = prepare_custom_config_dict['preserve_attr'][submodule_name]
            for attr_name in preserve_attr_list:
                logger.info("Preserve attr: {}.{}".format(submodule_name, attr_name))
                _attr = preserve_attr_dict[submodule_name][attr_name]
                if inspect.ismethod(_attr):
                    _attr = types.MethodType(getattr(_type, attr_name), cur_module)
                setattr(cur_module, attr_name, _attr)
    return prepared
