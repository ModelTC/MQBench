Naive PTQ
=========

MQBench provides a simple API for naive PTQ, learn our step-by-step instructions to quantize your model.

**1**. **To begin with, let's import MQBench and prepare FP32 model.**

.. code-block:: python

    import torchvision.models as models                           # for example model
    from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
    from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
    from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
    from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
    from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy

    model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
    model.eval()

**2**. **Choose your backend.**

.. code-block:: python

    # backend options
    backend = BackendType.Tensorrt
    # backend = BackendType.SNPE
    # backend = BackendType.PPLW8A16
    # backend = BackendType.NNIE
    # backend = BackendType.Vitis
    # backend = BackendType.ONNX_QNN
    # backend = BackendType.PPLCUDA
    # backend = BackendType.OPENVINO
    # backend = BackendType.Tengine_u8
    # backend = BackendType.Tensorrt_NLP

**3**. **The next step prepares to quantize the model.**

.. code-block:: python

    model = prepare_by_platform(model, backend)                   #! line 1. trace model and add quant nodes for model on backend
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

**4**. **Export quantized model.**

.. code-block:: python

    # define dummy data for model export.
    input_shape={'data': [10, 3, 224, 224]}
    convert_deploy(model, backend, input_shape)                   #! line 4. remove quant nodes, ready for deploying to real-world hardware

Now you know how to conduct naive PTQ with MQBench, if you want to know more about customize backend check :doc:`../internal/learn_config`.