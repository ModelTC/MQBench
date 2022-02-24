Vitis
=========

`Xilinx Vitis-AI <https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html>`_ is a platform for high-performance deep learning compression and inference on Xilinx FPGA device.

.. _Vitis Quantization Scheme:

Quantization Scheme
--------------------
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

Supported Operations
--------------------

Vitis-AI provides a software stack for DPUs of Xilinx FPGA, including network quantization and DPU inference etc.. To apply MQBench to Vitis-AI, we support a backend and its related observer and fake quantizer which is proven to be aligned with DPU. For now, we have supported typical operations used in classification and detection jobs: The supported Operations are listed here:

- Conv2d 
- ReLU
- MaxPooling2d
- Concat
- Input
- Add
- Resize(Interpolate)
- Flatten
- Gemm(Linear)
- GlobalAveragePool