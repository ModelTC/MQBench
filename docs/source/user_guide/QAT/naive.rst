Naive QAT
=========

The quantization aware training only requires some additional operations compared to ordinary fine-tune.

**1**. **Prepare FP32 model firstly.**

.. code-block:: python

    import torchvision.models as models
    from mqbench.convert_deploy import convert_deploy
    from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)
    model.train()

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

**3**. **Prepares to quantize the model.**

.. code-block:: python

    # trace model and add quant nodes for model on backend
    model = prepare_by_platform(model, backend)

    # calibration loop
    model.eval()
    enable_calibration(model)
    for i, batch in enumerate(data):
        # do forward procedures
        ...

    # training loop
    model.train()
    enable_quantization(model)
    for i, batch in enumerate(data):
        # do forward procedures
        ...

        # do backward and optimization
        ...

**4**. **Export quantized model.**

.. code-block:: python

    # define dummy data for model export.
    input_shape={'data': [10, 3, 224, 224]}
    convert_deploy(model, backend, input_shape)

Now you know how to conduct naive QAT with MQBench, if you want to know more about customize backend check :doc:`../internal/learn_config`.