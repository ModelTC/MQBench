SNPE
=========

`Snapdragon Neural Processing Engine (SNPE) <https://developer.qualcomm.com/sites/default/files/docs/snpe//index.html/>`_ is a Qualcomm Snapdragon software accelerated runtime for the execution of deep neural networks.

.. _SNPE Quantization Scheme:

Quantization Scheme
--------------------
8/16 bit per-layer asymmetric linear quantization.

.. math::

    \begin{equation}
        q = \mathtt{clamp}\left(\left\lfloor R * \dfrac{x - cmin}{cmax - cmin} \right\rceil, lb, ub\right)
    \end{equation}

where :math:`R` is the integer range after quantization, :math:`cmax` and :math:`cmin` are calculated range of the floating values, :math:`lb` and :math:`ub` are bounds of integer range.
Taking 8bit as an example, R=255, [lb, ub]=[0,255].


In fact, when building the SNPE with the official tools, it will firstly convert the model into *.dlc* model file of full precision, and then optionally change it into a quantized version.

.. attention::
    Users can provide a .json file to override the parameters.

    The values of *scale* and *offset* are not required, but can be overrided.

    SNPE will adjust the values of *cmin* and *cmax* to ensure zero is representable.
