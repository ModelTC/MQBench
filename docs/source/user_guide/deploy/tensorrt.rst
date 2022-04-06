TensorRT
========

Introduction
^^^^^^^^^^^^

`NVIDIA TensorRT <https://developer.nvidia.com/tensorrt>`_ is a platform for high-performance deep learning inference on GPU device.

.. _TensorRT Quantization Scheme:

**Quantization Scheme**

8bit per-channel symmetric linear quantization.

.. math::

    \begin{equation}
        q = \mathtt{clamp}(\lfloor x * s \rceil, lb, ub)
    \end{equation}

where :math:`s` is scaling factor to quantize a number from floating range to integer range, :math:`lb` and :math:`ub` are bounds of integer range.
For weights, [lb, ub] = [-127, 127]. For activations, [lb, ub] = [-128, 127].

For weights, each filter needs an independent scale :math:`s`.

Deploy on TensorRT
^^^^^^^^^^^^^^^^^^

**Requirements**:

- Install TensorRT>=8.0EA from `NVIDIA <https://developer.nvidia.com/tensorrt/>`_

**Deployment**:

We provide the example to deploy the quantized model to TensorRT using AdaRound and explicit mode.

- First edit </path-of-MQBench/application/imagenet_example/PTQ/configs/adaround/r18_8_8_trt.yaml>'s datasets, pretrained and output path, then export the quantized model to onnx.

    .. code-block:: shell
        :linenos:

        cd /path-of-MQBench/application/imagenet_example/PTQ/ptq
        python ptq.py --config /path-of-MQBench/application/imagenet_example/PTQ/configs/adaround/r18_8_8_trt.yaml

- Second build the TensorRT INT8 engine and evaluate, please make sure [dataset_path] contains subfolder [val].

    .. code-block:: shell
        :linenos:

        python onnx2trt.py --onnx <path-of-onnx_quantized_deploy_model.onnx> --trt <model_name.trt> --data <dataset_path> --evaluate
