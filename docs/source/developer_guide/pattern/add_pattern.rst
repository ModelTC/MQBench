Deal with New Fusion Pattern 
============================


What is Pattern in Quantization
-------------------------------

In quantization, there are **patterns** of fusion, which match certain pairs of float modules and turn them into **fused modules** [1]_. Normally, we apply certain patterns to our models like:

1. Conv2d + BN2d + ReLU
2. Conv2d + ReLU
3. Conv2d + BN2d 
4. Linear + ReLU 

After fusion, modules should be convert to QAT modules based on **mappings** to get the right gradients in finetune. It will map fused modules to **qat modules**. Fused modules have to give the right forward, while qat ones have to give the right forward and backward. Let's take a look at ConvBnReLU2d. We need to align its behavior to hardware, which merges BN into Conv2d. Fused module performs Conv2d, BN and ReLU in order, but the deployed module performs a Conv2d(fused) and ReLU. The quantization infomation should be about fused Conv2d rather than Conv2d and BN independently. Besides, the BN parameters should be updated. The code will be like:

.. code-block:: python
    :linenos:

    class FusedCBR2d(nn.Sequential):
        def __init__(self, conv, bn, relu):
            super().__init__(conv, bn, relu)
    
    class QATCBR2d(nn.Conv2d):
        ...
        def forward(self, x):
            running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            weight_shape = [1] * len(self.weight.shape)
            weight_shape[0] = -1
            bias_shape = [1] * len(self.weight.shape)
            bias_shape[1] = -1
            scaled_weight = self.weight_fake_quant(self.weight * scale_factor.reshape(weight_shape))
            if self.bias is not None:
                zero_bias = torch.zeros_like(self.bias)
            else:
                zero_bias = torch.zeros(self.out_channels, device=scaled_weight.device)
            conv = self._conv_forward(input, scaled_weight, zero_bias)
            conv_orig = conv / scale_factor.reshape(bias_shape)
            if self.bias is not None:
                conv_orig = conv_orig + self.bias.reshape(bias_shape)
            conv = self.bn(conv_orig)
            return conv

Relationship between qnn, qnni, qnnqat, qnniqat.
------------------------------------------------

Feel free to treat MQBench as an extension pack of PyTorch. The first alphabet 'q' stands for MQBench. 

1. ``nn``: float standalone modules.
2. ``nni``: float combined modules, which could be quantized into a union later.
3. ``nnqat``: quantized standalone modules.
4. ``nniqat``: quantized combined modules.

For developping new fuse patterns, we need implement the 2-4 in MQBench.


Add QAT Modules
---------------------

Of course, the very first step is to imply the standalone QAT modules like Conv2d or Linear. This will enable the quantization forward/backward simulation in the training and inferring. At ``mqbencn.nn.qat.modules``, you can implement the needed QAT modules based on its original function by inserting fake quantize nodes for weight, bias, activation or anything you want to.

Add Intrinsic Modules
---------------------

Intrinsic modules is more near to the deployed models, for it simulates the behavior in the platform by performing BN/ReLU merging and so on. Intrinsic modules wrap sub-modules into it.

First add a wrap module inheriting ``_FusedModule`` into ``mqbench.nn.intrinsic.modules.fused``. Actually, ``_FusedModule`` is an alias of ``nn.Sequential``, so it remains a float model and will not affect the function.

Then, we have to turn the fused float modules into quantized ones. At ``mqbench.nn.intrinsic.qat.modules``, implement the fused modules' QAT modules which load the parameters from the float ones and perform proper forward/backward(quantization and bn update) like it does in the platform. To be compatiable with the torch's API ``_fuse_fx``, we need to implement a classmethod ``from_float``. It will load all the parameters from float modules. 


TORCH Related infomation
------------------------

Torch has deployed its fusion pattern at [2]_, which could be applied in MQBench directly.


.. [1] https://github.com/pytorch/pytorch/blob/9cb52327a867cfc7878caf639a31fa5c860803a6/torch/ao/quantization/pattern.md
.. [2] https://github.com/pytorch/pytorch/blob/07932e27356c32df8a3c17361e4779c635d2d8ce/torch/ao/quantization/quantization_mappings.py#L36
