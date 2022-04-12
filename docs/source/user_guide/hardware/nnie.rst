NNIE
====

NNIE is a Neural Network Inference Engine of Hisilicon. It support INT8/INT16 quantization.

.. _NNIE Quantization Scheme:

Quantization Scheme
---------------------
8/16 bit per-layer logarithmic quantization.

The specific quantization formulation is:

.. math::

    \begin{equation}
    \begin{aligned}
        &z = \lfloor 16 * \log_2(c) \rceil - 127 \\
        &\mathtt{fakequant(x)} = \begin{cases}
            - 2 ^ {\dfrac{\mathtt{clamp}(\lfloor 16 * \log_2(-x) \rceil - z, 1, 127) + z}{16}}, & x \lt - 2 ^ {\dfrac{z + 1}{16} - 1} \\
            % 0, & - 2 ^ {\dfrac{z + 1}{16} - 1} \le x \lt 2 ^ {\dfrac{z}{16} - 1} \\
            2 ^ {\dfrac{\mathtt{clamp}(\lfloor 16 * \log_2(x) \rceil - z, 0, 127) + z}{16}}, & x \ge 2 ^ {\dfrac{z}{16} - 1} \\
            zero, & otherwise
        \end{cases}
    \end{aligned}
    \end{equation}

where :math:`c` is clipping range. :math:`2 ^ {\dfrac{z}{16}}` is the smallest positive value that can be represented after quantization.

It represents the integer number in *True Form* format.
The highest bit represents the sign and the rest represents the absolute value of the number.

.. list-table::
   :header-rows: 1
   :align: center

   * - Floating Numer
     - Integer Number
     - Hexadecimal
     - Dequantized Floating Number
   * - :math:`\bigg(- \infty, - 2 ^ {\dfrac{z + 126.5}{16}}\bigg]`
     - -127
     - 0xFF
     - :math:`- 2 ^ {\dfrac{z+127}{16}}`
   * - ...
     - ...
     - ...
     - ...
   * - :math:`\bigg(- 2 ^ {\dfrac{z + 2.5}{16}}, - 2 ^ {\dfrac{z + 1.5}{16}}\bigg]`
     - -2
     - 0x82
     - :math:`- 2 ^ {\dfrac{z+2}{16}}`
   * - :math:`\bigg(- 2 ^ {\dfrac{z + 1.5}{16}}, - 2 ^ {\dfrac{z + 1}{16} - 1}\bigg)`
     - -1
     - 0x81
     - :math:`- 2 ^ {\dfrac{z+1}{16}}`
   * - :math:`\bigg[- 2 ^ {\dfrac{z + 1}{16} - 1}, 2 ^ {\dfrac{z}{16} - 1}\bigg)`
     - -0
     - 0x80
     - 0
   * - :math:`\bigg[2 ^ {\dfrac{z}{16} - 1}, 2 ^ {\dfrac{z + 0.5}{16}}\bigg)`
     - 0
     - 0x00
     - :math:`2 ^ {\dfrac{z}{16}}`
   * - :math:`\bigg[2 ^ {\dfrac{z + 0.5}{16}}, 2 ^ {\dfrac{z + 1.5}{16}}\bigg)`
     - 1
     - 0x01
     - :math:`2 ^ {\dfrac{z+1}{16}}`
   * - ...
     - ...
     - ...
     - ...
   * - :math:`\bigg[2 ^ {\dfrac{z + 126.5}{16}}, + \infty\bigg)`
     - 127
     - 0x7F
     - :math:`2 ^ {\dfrac{z+127}{16}}`

NNIE performs a per-layer quantization, which means the inputs of the same layer share the same :math:`z_a` and the weights of the same layer share the same :math:`z_w`.

In fact, when building engine using the official tool of NNIE, it requires the clipping value :math:`c` rather than :math:`z`. :math:`c` needs to be a number in the 'gfpq_param_table_8bit.txt' which ensures that :math:`16 * \log_2{c}` is an integer.

.. attention::
    Pooling: ceil_mode = True

    Avoid using depthwise convolution.

    Only support 2x nearest neighbor upsample.

    For Detection task, you'd better choose RetinaNet structure.
