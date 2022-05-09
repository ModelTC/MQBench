Naive QAT
============

The training only requires some additional operations compared to ordinary fine-tune.

.. code-block:: python
    :linenos:

    import torchvision.models as models
    from mqbench.convert_deploy import convert_deploy
    from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)
    model.train()

    # then, we will trace the original model using torch.fx and \
    # insert fake quantize nodes according to different hardware backends (e.g. TensorRT).
    model = prepare_by_platform(model, BackendType.Tensorrt)

    # before training, we recommend to enable observers for calibration in several batches, and then enable quantization.
    model.eval()
    enable_calibration(model)

    # calibration loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

    model.train()
    enable_quantization(model)
    # training loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

        # do backward and optimization
        ...

    # deploy model, remove fake quantize nodes and dump quantization params like clip ranges.
    convert_deploy(model.eval(), BackendType.Tensorrt, input_shape_dict={'data': [10, 3, 224, 224]})

Now you know how to conduct naive QAT with MQBench, if you want to know more about customize backend check :doc:`../internal/learn_config`.