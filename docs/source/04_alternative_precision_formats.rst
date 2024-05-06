Alternative Precision Format
=============================================

FP8 quantization
-------------------------------

Sophgo-mq supports quantizing weights and activations in FP8 format.

Below is a code example of performing FP8 quantization using the E4M3 format:

.. code:: python

  prepare_custom_config_dict = {
        'quant_dict': {
                        'chip': 'BM1690',
                        'quantmode': 'weight_activation',
                        'strategy': 'CNN',
                       },
        'extra_qconfig_dict': {
                                'w_observer': 'MinMaxObserver',
                                'a_observer': 'EMAMinMaxObserver',
                                "w_fakequantize": 'E4M3FakeQuantize',
                                "a_fakequantize": 'E4M3FakeQuantize',
                                'w_qscheme': {  'bit': 8,
                                                'symmetry': True,
                                                'per_channel': False,
                                                'pot_scale': False },
                                'a_qscheme': {  'bit': 8,
                                                'symmetry': True,
                                                'per_channel': False,
                                                'pot_scale': False }
                              }
        }

  model = prepare_by_platform(model, prepare_custom_config_dict)

Below is a code example of performing FP8 quantization using the E5M2 format:

.. code:: python

  prepare_custom_config_dict = {
        'quant_dict': {
                        'chip': 'BM1690',
                        'quantmode': 'weight_activation',
                        'strategy': 'CNN',
                       },
        'extra_qconfig_dict': {
                                'w_observer': 'MinMaxObserver',
                                'a_observer': 'EMAMinMaxObserver',
                                "w_fakequantize": 'E5M2FakeQuantize',
                                "a_fakequantize": 'E5M2FakeQuantize',
                                'w_qscheme': {  'bit': 8,
                                                'symmetry': True,
                                                'per_channel': False,
                                                'pot_scale': False },
                                'a_qscheme': {  'bit': 8,
                                                'symmetry': True,
                                                'per_channel': False,
                                                'pot_scale': False }
                              }
        }

  model = prepare_by_platform(model, prepare_custom_config_dict)
