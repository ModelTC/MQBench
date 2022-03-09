Vitis
========

Introduction
^^^^^^^^^^^^

`Xilinx Vitis <https://github.com/Xilinx/Vitis-AI/>`_ is a platform for high-performance deep learning inference on Xilinx FPGA device.

.. _Vitis Quantization Scheme:

**Quantization Scheme**

8bit per-tensor symmetric linear quantization with power of two scales.

For weights/biases:

.. math::

    \begin{equation}
        q = \mathtt{clamp}(\lfloor x * s \rceil, lb, ub)
    \end{equation}

For activations:

.. math::

    \begin{equation}
        \begin{aligned}
            q &= \mathtt{clamp}(\lceil x * s \rceil, lb, ub), \text{ where } x*s-\lfloor x*s\rfloor = 0.5 \text{ and } x < 0 \\
            q &= \mathtt{clamp}(\lfloor x * s \rceil, lb, ub), \text{ else}.
        \end{aligned}
    \end{equation}


where :math:`s` is scaling factor to quantize a number from floating range to integer range, :math:`lb` and :math:`ub` are bounds of integer range, and [lb, ub] = [-128, 127].

Deploy on Vitis
^^^^^^^^^^^^^^^^^^

**Deployment**:

We provide the example to deploy the quantized `EOD <https://github.com/ModelTC/EOD>`_ model to Vitis, which is winner solution for the Low Power Computer Vision Challenge 2021 (`LPCV2021 <https://github.com/ModelTC/LPCV2021_Winner_Solution>`_).

- First quantize model in EOD.
    
    .. code-block:: shell
        :linenos:

        python -m eod train -e --config configs/det/yolox/yolox_fpga_quant_vitis.yaml --nm 1 --ng 1 --launch pytorch 2>&1 | tee log_qat_mqbench


- Second export the quantized model to ONNX [mqbench_qmodel.onnx] and [mqbench_qmodel_deploy_model.onnx].
    
    .. code-block:: shell
        :linenos:

        python -m eod quant_deploy --config configs/det/yolox/yolox_fpga_quant_vitis.yaml --ckpt [model_save_path] --input_shape [input_shape] 2>&1 | tee log.delpoy.txt

- Third build Docker from `Dockerfile <https://github.com/ModelTC/MQBench/tree/main/docker>`_, convert ONNX to xmodel [mqbench_qmodel_deploy_model.onnx_int.xmodel].

    .. code-block:: shell
        :linenos:

        python -m mq.dep.convert_xir -Q [mqbench_qmodel.onnx] -C [mqbench_qmodel_deploy_model.onnx] -N [model_name]

- Fourth compile xmodel to deployable model [mqbench_qmodel.xmodel].

    .. code-block:: shell
        :linenos:

        vai_c_xir -x [mqbench_qmodel_deploy_model.onnx_int.xmodel] -a [new_arch.json] -o [output_path] -n [model_name]
