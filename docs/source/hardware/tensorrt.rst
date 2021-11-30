TensorRT
=========

`NVIDIA TensorRT <https://developer.nvidia.com/tensorrt>`_ is a platform for high-performance deep learning inference on GPU device.

.. _TensorRT Quantization Scheme:

Quantization Scheme
--------------------
8bit per-channel symmetric linear quantization.

.. math::

    \begin{equation}
        q = \mathtt{clamp}(\lfloor x * s \rceil, lb, ub)
    \end{equation}

where :math:`s` is scaling factor to quantize a number from floating range to integer range, :math:`lb` and :math:`ub` are bounds of integer range.
For weights, [lb, ub] = [-127, 127]. For activations, [lb, ub] = [-128, 127].

For weights, each filter needs an independent scale :math:`s`.

In fact, when building the TensorRT engine, the official tool requires the clipping value as quantization parameters, which can be calculated by :math:`c = s * 127`.
