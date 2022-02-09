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
        # benchmark test 
        benchmark_app -m ./openvino_deploy_model.xml
