Learn MQBench configuration
===========================

MQBench provides a primary API **prepare_by_platform** for users to quantize their model. 
MQBench contains many backends presets for **hardware alignment**, but you maybe want to customize your backend.
We provide a guide for learning MQBench configuration, and it will be helpful.

**1.** API **prepare_by_platform** accepts an extra param, you can provide it following this format.

.. code-block:: python

    extra_config = {
        'w_observer': MSEObserver,                                # custom weight observer
        'a_observer': MSEObserver,                                # custom activation observer
        'w_fakequantize': FixedFakeQuantize,                      # custom weight fake quantize function
        'a_fakequantize': FixedFakeQuantize,                      # custom activation fake quantize function
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
    }


**2.** **Customize just by:**

.. code-block:: python

    prepared = prepare_by_platform(model, backend, extra_config)

**3.** **Now MQBench support this Observers and Quantizers**

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