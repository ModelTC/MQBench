Naive PTQ
=========

MQBench provides a simple API for naive PTQ, learn our step-by-step instructions to quantize your model. You can also see :doc:`../../get_started/quick_start_academic` for more details.

.. code-block:: python
    :linenos:

    import torchvision.models as models                           # PyTorch model
    from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
    from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
    from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
    from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8

    model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
    model.eval()

    model = prepare_by_platform(model, BackendType.Tensorrt)      #! line 1. trace model and add quant nodes for model on Tensorrt Backend
    enable_calibration(model)                                     #! line 2. turn on calibration, ready for gathering data

    # calibration loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

    enable_quantization(model)                                    #! line 3. turn on actually quantization, ready for simulating Backend inference

    # evaluation loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

Now you know how to conduct naive PTQ with MQBench, if you want to know more about customize backend check :doc:`../internal/learn_config`.