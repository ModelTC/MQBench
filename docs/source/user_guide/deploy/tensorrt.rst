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

- Install TensorRT=7.2.1.6 from `NVIDIA <https://developer.nvidia.com/tensorrt/>`_

**Deployment**:

We provide the example to deploy the quantized model to TensorRT.

- First export the quantized model to ONNX [tensorrt_deploy_model.onnx] and dump the clip ranges [tensorrt_clip_ranges.json] for activations.

    .. code-block:: shell
        :linenos:

        python main.py -a [model_name] --resume [model_save_path]

- Second build the TensorRT INT8 engine and evaluate, please make sure [dataset_path] contains subfolder [val].

    .. code-block:: shell
        :linenos:

        python onnx2trt.py --onnx [tensorrt_deploy_model.onnx] --trt [model_name.trt] --clip [tensorrt_clip_ranges.json] --data [dataset_path] --evaluate

- If you don’t pass in external clip ranges [tensorrt_clip_ranges.json], TenosrRT will do calibration using default algorithm IInt8EntropyCalibrator2 with 100 images. So, please make sure [dataset_path] contains subfolder [cali].

    .. code-block:: shell
        :linenos:

        python onnx2trt.py --onnx [tensorrt_deploy_model.onnx] --trt [model_name.trt] --data [dataset_path] --evaluate
