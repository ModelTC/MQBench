Advanced PTQ
============

MQBench provides a simple API for advanced PTQ, learn our step-by-step instructions to quantize your model.

**1**. **Prepare FP32 model firstly.**

.. code-block:: python

    import torchvision.models as models
    from mqbench.convert_deploy import convert_deploy
    from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization
    from mqbench.advanced_ptq import ptq_reconstruction

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)
    model.eval()

**2**. **Configure advanced ptq and backend.**

.. code-block:: python

    # configuration
    ptq_reconstruction_config = {
        'pattern': 'block',                   #? 'layer' for Adaround or 'block' for BRECQ and QDROP
        'scale_lr': 4.0e-5,                   #? learning rate for learning step size of activation
        'warm_up': 0.2,                       #? 0.2 * max_count iters without regularization to floor or ceil
        'weight': 0.01,                       #? loss weight for regularization item
        'max_count': 20000,                   #? optimization iteration
        'b_range': [20,2],                    #? beta decaying range
        'keep_gpu': True,                     #? calibration data restore in gpu or cpu
        'round_mode': 'learned_hard_sigmoid', #? ways to reconstruct the weight, currently only support learned_hard_sigmoid
        'prob': 1.0,                          #? dropping probability of QDROP, 1.0 for Adaround and BRECQ
    }

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

**3**. **Prepare to quantize the model.**

.. code-block:: python

    # trace model and add quant nodes for model on backend
    model = prepare_by_platform(model, backend)

    # calibration loop
    enable_calibration(model)
    for i, batch in enumerate(data):
        # do forward procedures
        ...

    # ptq_reconstruction loop
    stacked_tensor = []
    # add ptq_reconstruction data to stack
    for i, batch_data in enumerate(data):
        if i == cali_batchsize:
            break
        stacked_tensor.append(batch_data)
    # start ptq_reconstruction
    model = ptq_reconstruction(model, stacked_tensor, ptq_reconstruction_config)

    # evaluation loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

**4**. **Export quantized model.**

.. code-block:: python

    # deploy model, remove fake quantize nodes, and dump quantization params like clip ranges.
    input_shape={'data': [10, 3, 224, 224]}
    convert_deploy(model, backend, input_shape)

you can find algorithm details in :doc:`../algorithm/advanced_ptq`. We also provides an example in `here <https://github.com/ModelTC/MQBench/tree/main/application/imagenet_example/PTQ/>`_.
