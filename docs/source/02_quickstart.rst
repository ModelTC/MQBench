Quick Start
=====================

In this section, we use the PTQ INT8 quantization of ResNet-18 as an example to understand the working principles of Sophgo-mq. Let's begin step by step.

Import packages and define the model.
--------------------------------------------------------------

First, you need a model that has been pre-trained with PyTorch; here we choose ResNet18.

.. code-block:: python

  import torch
  import torchvision.models as models
  from sophgo_mq.prepare_by_platform import prepare_by_platform
  from sophgo_mq.convert_deploy import convert_deploy
  from sophgo_mq.utils.state import enable_quantization, enable_calibration

  model = models.__dict__['resnet18']()



Use prepare_by_platform function to insert fake quantization nodes.
---------------------------------------------------------------------------------------------

The function **prepare_by_platform** is used to trace the computational graph of the model using **torch.fx**, 
and to insert fake quantization nodes at the appropriate positions.

When quantizing a model, it is necessary to specify the chip type, quantization mode, and quantization strategy.

- The chip type determines the precision formats supported by the quantization.
- When the quantization mode is 'weight_activation', both weights and activations are quantized; when the mode is 'weight_only', only the model's weights are quantized.
- When the model is a Convolutional Neural Network, the quantization strategy should be 'CNN'; when the model belongs to the transformer series, the quantization strategy should be 'Transformer'.

.. code-block:: python

  extra_prepare_dict = {
      'quant_dict': {
                      'chip': 'BM1690',
                      'quantmode': 'weight_activation',
                      'strategy': 'CNN',
                      },
  }
  model = prepare_by_platform(model, prepare_custom_config_dict=extra_prepare_dict)
  model.eval()



Define your dataloader and perform forward propagation.
---------------------------------------------------------------------------------------------

The function **enable_calibration** serves to enable observer for gathering statistical distribution data of weights or activations, 
and subsequently computes the quantization information.

The function **enable_quantization** disables observer and enables quantize. 
When a model performs forward inference, the model's weights and activations will be quantized.

In the following code, we first use the function **enable_calibration** to enable observer, 
and then perform forward propagation with a number of data samples. 
During propagation, the observer collects statistical information for quantization. 
Finally, we use the **enable_quantization** function to enable quantize.

.. code-block:: python

  dataloder = ...
  enable_calibration(model) ## enable observer to gather calibration information
  for input in dataloader:
      output = model(input)
  enable_quantization(model) ## enable quantize



Use convert_deploy function to export Onnx file, calitable and qtable. 
---------------------------------------------------------------------------------------------

.. code-block:: python

  convert_deploy(model, net_type='CNN', 
                  input_shape_dict={'input': [1, 3, 224, 224]}, 
                  output_path='./', 
                  model_name='resnet18')
