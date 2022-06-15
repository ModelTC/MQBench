Learn MQBench configuration
===========================

Configuration
^^^^^^^^^^^^^

MQBench provides a primary API **prepare_by_platform** for users to quantize their model. 
MQBench contains many backends presets for **hardware alignment**, but you maybe want to customize your backend.
We provide a guide for learning MQBench configuration, and it will be helpful.

**1.** API **prepare_by_platform** accepts an extra param, you can provide it following this format.

.. code-block:: python

    extra_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',                              # custom weight observer
            'a_observer': 'MSEObserver',                              # custom activation observer
            'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
            'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
            'w_qscheme': {
                'bit': 8,                                             # custom bitwidth for weight,
                'symmetry': False,                                    # custom whether quant is symmetric for weight,
                'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for weight,
                'pot_scale': False,                                   # custom whether scale is power of two for weight.
            },
            'a_qscheme': {
                'bit': 8,                                             # custom bitwidth for activation,
                'symmetry': False,                                    # custom whether quant is symmetric for activation,
                'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for activation,
                'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        },
        'extra_quantizer_dict': {
            'additional_function_type': [operator.add,],              # additional function type, a list, use function full name, like operator.add.
            'additional_module_type': (torch.nn.Upsample),            # additional module type, a tuple, use class full name, like torch.nn.Upsample.
            'additional_node_name': [layer1_1_conv1] ,                # addition node name, a list, use full node name, like layer1_1_conv1.
            'exclude_module_name': [layer2.1.relu,],                  # skip specific module, a list, use module qualify name, like layer2.1.relu.
            'exclude_function_type': [operator.mul,] ,                # skip specific module, a list, use function full name, like operator.mul
            'exclude_node_name': [layer1_1_conv1],                    # skip specific module, a list, use full node name, like layer1_1_conv1.
        },
        'preserve_attr': {
            '': ["func1"],                                            # Specify attribute of model which should be preserved
            'backbone': ['func2'],                                    # after prepare. Since symbolic_trace only store attributes which is
                                                                      # in forward. If model.func1 and model.backbone.func2 should be preserved,
                                                                      # {'': ['func1'], 'backbone': ['func2'] } should work.
        }
        'extra_fuse_dict': {                                            # checkout https://github.com/ModelTC/MQBench/blob/main/mqbench/fuser_method_mappings.py for more fuse details.
            'additional_fuser_method_mapping': {
                (torch.nn.Linear, torch.nn.BatchNorm1d):
                    fuse_linear_bn,                                   # fuse use method

            },
            'additional_fusion_pattern': {
                (torch.nn.BatchNorm1d, torch.nn.Linear):
                    ConvBNReLUFusion,                                 # fuse use pattern
            },
            'additional_qat_module_mapping': {
                nn.ConvTranspose2d: qnn.qat.ConvTranspose2d,          # mapping qat module
            } ,
        },
        'concrect_args': {
        }                                                             # custom tracer behavior, checkout https://github.com/pytorch/pytorch/blob/efcbbb177eacdacda80b94ad4ce34b9ed6cf687a/torch/fx/_symbolic_trace.py#L836
    }


**2.** **Customize just by:**

.. code-block:: python

    prepared = prepare_by_platform(model, backend, extra_config)

Observer
^^^^^^^^

.. code-block:: markdown

    1. MinMaxObserver
    2. EMAMinMaxObserver        # More general choice
    3. MinMaxFloorObserver      # For Vitis HW
    4. EMAMinMaxFloorObserver   # For Vitis HW
    5. EMAQuantileObserver      # Quantile observer.
    6. ClipStdObserver          # Usually used for DSQ.
    7. LSQObserver              # Usually used for LSQ.
    8. MSEObserver
    9. EMAMSEObserver

Quantizer
^^^^^^^^^
.. code-block:: markdown

    1. FixedFakeQuantize        # Unlearnable scale/zeropoint
    2. LearnableFakeQuantize    # Learnable scale/zeropoint
    3. NNIEFakeQuantize         # Quantize function for NNIE
    4. DoReFaFakeQuantize       # Dorefa
    5. DSQFakeQuantize          # DSQ
    6. PACTFakeQuantize         # PACT
    7. TqtFakeQuantize          # TQT
    8. AdaRoundFakeQuantize