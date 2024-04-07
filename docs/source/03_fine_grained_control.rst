Fine-grained Control
=====================


In this section, we introduce how to control the quantization process with fine granularity, 
such as specifying the quantization method for a certain module within the network, 
determining which layers of the network should not be quantized, and so on.

Overview
-------------------------------


In order to achieve fine-grained control over the model quantization process, 
we can provide certain dictionaries to dictate specific quantization actions when employing prepare_by_platform. 
These include:

- **quant_dict** : specify the chip type, quantization mode, and quantization strategy.
- **extra_qconfig_dict** : specify the qconfig for each module.
- **extra_qconfig_dict** : specify the behavior of the quantizer.

In the following content, we will provide explanations of the above three dictionaries, as well as their specific usage.

.. code-block:: python

  prepare_custom_config_dict = {
                                  'quant_dict': quant_dict 
                                  'extra_qconfig_dict': extra_qconfig_dict 
                                  'extra_qconfig_dict': extra_qconfig_dict 
                              }

  model = prepare_by_platform(model, prepare_custom_config_dict)


Quant_dict
-------------------------------

Quant_dict is used to specify the chip type, quantization mode, and quantization strategy.

The specific definition is as follows:

.. code-block:: python

  quant_dict = {
                  'chip': 'SG2260', # ['BM1688', 'BM1684X', 'SG2260', 'Academic']
                  'quantmode': 'weight_activation', # ['weight_only', 'weight_activation'] 
                  'strategy': 'CNN', # ['CNN', 'Transformer']
              }




Extra_qconfig_dict
-------------------------------

The Extra_qconfig_dict is used to specify the qconfig for global use, as well as the qconfig for a certain type of module within the network. 
Furthermore, the qconfig can also be designated by using the name of a specific module.


Specify the qconfig for certain module type. 
Here, we specify different qconfigs for the **qnniqat.ConvBnReLU2d_sophgo** class and the **qnniqat.ConvReLU2d_sophgo** class, respectively.

.. code-block:: python

  object_type = {
                  qnniqat.ConvBnReLU2d_sophgo: {  # qconfig for nniqat.ConvBnReLU2d_sophgo
                              'mode': 'activation',
                              'bit': '8',
                              'afakequantize': 'LearnableFakeQuantize',
                              'aobserver': 'MinMaxObserver',
                            },
                  
                  qnniqat.ConvReLU2d_sophgo: {  # qconfig for qnniqat.ConvReLU2d_sophgo
                              'mode': 'weight',
                              'bit': '8',
                              'wfakequantize': 'FixedFakeQuantize',
                              'wobserver': 'MinMaxObserver',
                            },
                }


Specify the qconfig for specific module. Here, we use the name of the modules in the network (which can be obtained through **model.named_modules()**) to specify qconfig for a particular module.

.. code-block:: python

  module_name = {
                  'layer1.conv1': {   # qconfig for 'layer1.conv1' 
                              'mode': 'activation',
                              'bit': '8',
                              'afakequantize': 'LearnableFakeQuantize',
                              'aobserver': 'MinMaxObserver',
                            },
                  
                  'layer2.conv2': {  # qconfig for 'layer2.conv2'
                              'mode': 'weight',
                              'bit': '8',
                              'wfakequantize': 'FixedFakeQuantize',
                              'wobserver': 'MinMaxObserver',
                            },
                }

Use **extra_qconfig_dict** to define a global quantization configuration, and utilize the aforementioned **object_type** and **module_name**.

.. code-block:: python

  extra_qconfig_dict = {
                        'w_observer': 'MinMaxObserver', # global weight observer
                        'a_observer': 'EMAMinMaxObserver', # global activation observer
                        "w_fakequantize": 'E4M3FakeQuantize', # global weight fakequantize
                        "a_fakequantize": 'E4M3FakeQuantize', # global activation fakequantize
                        'w_qscheme': {  'bit': 8,  # global weight qscheme
                                        'symmetry': True,
                                        'per_channel': False,
                                        'pot_scale': False },
                        'a_qscheme': {  'bit': 8,  # global activation qscheme
                                        'symmetry': True,
                                        'per_channel': False,
                                        'pot_scale': False }
                        'object_type': object_type,
                        'module_name': module_name,
                     }



Extra_quantizer_dict
-------------------------------

The **extra_quantizer_dict** is used to specify the behavior of the quantizer, 
such as setting fake quantization nodes to only observe and not quantize, 
or to avoid inserting fake quantization nodes in certain modules or functions.

The purpose of the below **extra_quantizer_dict** is to:

- Specify that the fake quantization node named **features.0.0.weight_fake_quant** should only observe and not quantize.
- Prevent quantization of the weights for **layer3.conv3**, and avoid inserting activation fake quantization node before this module.
- Avoid inserting activation fake quantization node before the **torch.nn.functional.sigmoid** function.

.. code-block:: python

  extra_quantizer_dict = {
                          'module_only_enable_observer': ['features.0.0.weight_fake_quant'],
                          'exclude_module_name': ['layer3.conv3'],
                          'exclude_function_type': [torch.nn.functional.sigmoid],
                        }
