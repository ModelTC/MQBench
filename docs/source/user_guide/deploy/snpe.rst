SNPE
====

Introduction
^^^^^^^^^^^^

`Snapdragon Neural Processing Engine (SNPE) <https://developer.qualcomm.com/sites/default/files/docs/snpe//index.html/>`_ is a Qualcomm Snapdragon software accelerated runtime for the execution of deep neural networks.

.. _SNPE Quantization Scheme:

**Quantization Scheme**

8/16 bit per-layer asymmetric linear quantization.

.. math::

    \begin{equation}
        q = \mathtt{clamp}\left(\left\lfloor R * \dfrac{x - cmin}{cmax - cmin} \right\rceil, lb, ub\right)
    \end{equation}

where :math:`R` is the integer range after quantization, :math:`cmax` and :math:`cmin` are calculated range of the floating values, :math:`lb` and :math:`ub` are bounds of integer range.
Taking 8bit as an example, R=255, [lb, ub]=[0,255].


Deploy on SNPE
^^^^^^^^^^^^^^

**Requirements**:

- Install SNPE SDK from `QualComm <https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html>`_ (Suggest Ubuntu 18.04)

**Deployment**:

- Convert PyTorch checkpoint to `snpe_deploy.onnx` and dump clip ranges to `snpe_clip_ranges.json`:

    .. code-block:: python
        :linenos:

        from mqbench.convert_deploy import convert_deploy
        input_dict = {'x': [1, 3, 224, 224]}
        convert_deploy(solver.model.module, BackendType.SNPE, input_dict)

- Convert `.onnx` file to `.dlc` format (supported by SNPE):

    .. code-block:: shell
        :linenos:

        # Note that, the `.json` file contains activation ranges for quantization, but it's required here although the model hasn't been quantized now.
        snpe-onnx-to-dlc --input_network ./snpe_deploy.onnx --output_path ./snpe_deploy.dlc --quantization_overrides ./snpe_clip_ranges.json

- Quantize the model with parameters overridden:

    .. code-block:: shell
        :linenos:

        # The `data.txt` records paths to image data for calibration (not important since we will override parameters) which will be loaded by `numpy.fromfile(dtype=np.float32)` and have shape of `(224, 224, 3)`. And this file is required for test.
        # Now we get the final model `snpe_deploy_quantized.dlc`
        snpe-dlc-quantize --input_dlc ./snpe_deploy.dlc --input_list ./data.txt --override_params  --bias_bitwidth 32