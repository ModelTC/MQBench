A Brief about How MQBench Supports Hardware and Software 
======================================================

MQBench is able to do quantization over many different backends and quantization algorithms, which relys on the independence of hardware(backends) and software(algorithms). To support a new backend, h/w and s/w should be added.

We provide a typical work-flow:

1. Add ``FakeQuantizeAffine`` to simulate the hardware behavior.
2. Add corresponding ``Observer`` to provide ``FakeQuantizeAffine`` with needed infomation.
3. Add ``ModelQuantizer`` to insert quantize nodes into the model.
4. Add ONNX-Backend translator to deploy the output graph.



FakeQuantize, Observer and Quantization Scheme
----------------------------------------------

When the model is at calib mode, ``Observer`` collects needed statistics and ``FakeQuantize`` does not quantize the input; when at quant mode, ``FakeQuantize`` performs the quantized forward with the qparams calulated bt ``Observer``. 

To add a ``FakeQuantizeAffine``, refer to ``mqbench/fake_quantize/``. 


1. Fucntion: Add a fake quantization function to describe the quantized forward. If the function is not differential, also describe the backward.
2. Autograd and Symbolic: Wrap the quantization forward/backward function with ``torch.autograd.Function``. To enable onnx export, a ``symbolic`` function is also needed.
3. Inherit the QuantizeBase: Define the class which holds the ``qparams`` and performs the quantized forward. This class inherit ``mqbench.fake_quantize.quantize_base.QuantizeBase``.
4. Add a Observer if needed: Inherit ``ObserverBase``, collect statistics in ``forward`` and get qparams in ``calculate_qparams``.
5. Use it in prepare_by_platform

    i. Import your class at the ``mqbench.prepare_by_platform``. 
    ii. Add your backend type into the Enumeration ``mqbench.prepare_by_platform.BackendType``. 
    iii. Define its default **scheme** at ``mqbench.prepare_by_platform.ParamsTable``.
    iv. Add mappings to and  ``mqbench.prepare_by_platform.FakeQuantizeDict``.

For now, we have provided some linear quantization affines/observers so it might reduce lots of work.


Custom Quantizer
----------------

To prepare the model by platform, quantization nodes should be inserted by and **only by** ``ModelQuantizer``, which hides all details about the needed graph. After preparation, the model could be used to QAT or PTQ. 

The ModelQuantizer will do:

1. Op fusion, like Conv2d+BN2d+ReLU -> ConvBnReLU2d. 
2. QAT swap. Fuse the modules into QAT modules and insert quantize node for weights. 
3. Insert quantize nodes after activation. Need to find whose  output will be quantized in the backend. 

Op Fusion 
^^^^^^^^^

The fusion is a torch API and you need to edit the torch default fuser method mappings and patterns because the torch fuser does not accept any other args. 

If you add some new fusion patterns: 

1. Add the fusion patterns and related fusion method into  ``mqbench.fuser_method_mapping``. This will turn some op patterns into intrinsic fused modules. If you create new intrinsic modules, add them into ``mqbench.nn.intrinsic``. If you apply new fusion method, add them into ``mqbench.utils.fusion``.
2. Add the mapping from intrinsic modules into qat modules. Define the modules at ``mqbench.nn.intrinsic.qat`` if needed.
3. Add all the metioned things to ``mqbench.fuser_method_mapping.fuse_custom_config_dict``. 
4. **NOTE**: If the pattern only apply to certain backend, you need to update the default dicts at ModelQuantizer, rather than ``fuse_custom_config_dict``! 

QAT Swap
^^^^^^^^

Swap the intrinsic modules into qat ones. Add the mappings into  ``mqbench.fuser_method_mapping`` if it is a universal mapping. If not, into ModelQuantizer's additional_qat_module_mappings.

We deploy bias-quant intrinsic and qat modules, so just refer to the ``mqbench.nn`` to see how to add one by yourself.

Insert Quantize nodes
^^^^^^^^^^^^^^^^^^^^^^

Usually the TensorRT's ModelQuantizer is a good example, which quantizes all quantizable input tensors of module. If you need to quantize other nodes, just add them to the set with your own logic.


Deploy Functions
----------------

The deploy stage is to turn our torch-based model into ONNX and graphs based on backend. Typically, we will merge bn into conv/deconv/linear, convert the quantized model into onnx and finally remove all fake quantize op and deploy them. The functions are defined at ``mqbench.convert_deploy`` and regisered by ``@register_deploy_function(BackendType)``. 

Deploy stage might be the hardest part of the flow, for it takes care of all h/w details. The normal flow will be like:

1. No extra work needed for merging bn and conver onnx.
2. For removing fake_quantize and collecting qparams, you can check the ``mqbench.deploy``. There are examples for linear/logarithmic quantization and self-defined onnx runtime.
3. Also, there are platforms support standard onnx quantization. If your platform supports this, please export FakeQuantizeAffine into onnxruntime QDQOperators. [1]_ [2]_ 
4. If extra tools are needed to complete the translation, just integrate your tools into ``mqbench.deploy``. And there is also an example of Vitis-AI.

.. [1] https://onnxruntime.ai/docs/performance/quantization.html
.. [2] https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/quantization/registry.py
