OPENVINO
========


Introduction
^^^^^^^^^^^^

`OpenVINO™ <https://docs.openvino.ai/latest/documentation.html>`_ is an open-source toolkit for optimizing and deploying AI inference on a range of Intel® platforms from edge to cloud.

.. _OPENVINO Quantization Scheme:

**Quantization Scheme**

- Support of mixed-precision models where some layers can be kept in the floating-point precision.
- Per-channel quantization of weights of Convolutional and Fully-Connected layers.
- Per-channel quantization of activations for channel-wise and element-wise operations, e.g. Depthwise Convolution, Eltwise Add/Mul, ScaleShift.
- Symmetric and asymmetric quantization of weights and activations with the support of per-channel scales and zero-points.
- Non-unified quantization parameters for Eltwise and Concat operations.
- Non-quantized network output, i.e. there are no quantization parameters for it.

More details can be found at https://github.com/openvinotoolkit/nncf/blob/2f231aa3903a286dafaa15eaae54758e2a2f346b/docs/compression_algorithms/Quantization.md


Deploy on OpenVINO
^^^^^^^^^^^^^^^^^^

**Requirements**:

- Install OpenVINO C++ SDK from `Intel <https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html>`_
- Install OpenVINO Python SDK using command `pip install openvino openvino-dev` (optional)


**Deployment**:

- Python tutorials (see MQBench github `application/openvino_example.ipynb`) are written for running on jupyter notebooks, including PTQ process and accuracy evaluation.

- Convert PyTorch checkpoint to `openvino_deploy_model.onnx`:

    .. code-block:: python
        :linenos:

        from mqbench.convert_deploy import convert_deploy
        input_dict = {'x': [1, 3, 224, 224]}
        convert_deploy(model, BackendType.OPENVINO, input_dict, model_name = 'openvino')

- Convert `.onnx` file to `.xml` format and `.bin` format (supported by OpenVINO):

    .. code-block:: shell
        :linenos:

        # mo --help get more information or check the docs for openvino
        mo --input_model ./openvino_deploy_model.onnx
        # after exec prev line, you will get openvino_deploy_model.xml and openvino_deploy_model.bin
        # benchmark test using one cpu
        benchmark_app -m ./openvino_deploy_model.xml -nstream 1
        # test result on  Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz Model: resnet18
        # Original top-1 accuracy: 69.758
        # PTQ top-1  accuracy: 69.334
        # deploy using openvino top-1  accuracy: 69.312
        # cosine distance between torch model and openvino IR measured on last output:0.9975
        # Benchmark Result
        # Original Resnet18>> Count: 6959  iterations Duration: 60009.54 ms Latency: 8.71 ms Throughput: 115.96 FPS
        # Quantized Version>> Count: 13094 iterations Duration: 60004.75 ms Latency: 4.44 ms Throughput: 218.22 FPS
