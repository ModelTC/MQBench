Tengine
========


Introduction
^^^^^^^^^^^^

`Tengine <https://tengine-docs.readthedocs.io/en/latest/index.html>`_ is a lite, high performance, modular inference engine for embedded device, which powered by `OPEN AI LAB <http://www.openailab.com/>`_.

.. _Tengine Quantization Scheme:

**Quantization Scheme**

- Full 8-bit integer representation of model weights and computation.
- Per-tensor quantization of all weights and activations.
- Asymmetric quantization of all weights and activations in ``Tengine_u8`` mode.
- Quantization of input and de-quantization of output should be done manually.

More details can be found at https://github.com/OAID/Tengine


Deploy on Tengine
^^^^^^^^^^^^^^^^^^

**Requirements**:

- Compile and install Tengine toolkit from `Tengine <https://github.com/OAID/Tengine>`_
- Install Tengine Python API from `Pytengine <https://github.com/OAID/Tengine/tree/tengine-lite/pytengine>`_


**Deployment**:

We provide the example to deploy the quantized model to Tengine with `asymmetric quantization <https://tengine-docs.readthedocs.io/en/latest/user_guides/quant_tool_uint8.html>`_.

- First export the quantized model to ONNX [mqbench_qmodel_for_tengine.onnx] and dump the quantization parameters [mqbench_qmodel_for_tengine.scale] for activations.

    .. code-block:: shell
        :linenos:

        python main.py -a [model_name] --resume [model_save_path] --deploy --backend tengine_u8

- Second convert ``.onnx`` file into ``.tmfile`` format supported by Tengine (https://tengine-docs.readthedocs.io/en/latest/user_guides/convert_tool.html).

    .. code-block:: shell
        :linenos:

        tm_convert_tool -f onnx -m [mqbench_qmodel_for_tengine.onnx] -o [xxxx.tmfile]

- Quantize ``.tmfile`` with ``mqbench_qmodel_for_tengine.scale`` (ref: https://tengine-docs.readthedocs.io/en/latest/user_guides/quant_tool_uint8.html).

    .. code-block:: shell
        :linenos:

        quant_tool_uint8 -m [xxx.tmfile] -o [xxxx_u8.tmfile] -i ./ -f [mqbench_qmodel_for_tengine.scale]

- Validation with pytengine(optional).

    .. code-block:: shell
        :linenos:

        python eval_tengine.py  --dataset [path to dataset] -m [xxxx_u8.tmfile]
