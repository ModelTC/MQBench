Develop QAT with MQBench
================================

Given a model which has been prepared by ``mqbench.prepare_by_platform``, fake quantize nodes are accessible for further training. Then, we divide parameters into two groups: normal parameters and quantization parameters: 

.. code-block:: python
    :linenos:

    # SomeFakeQuantize
    normal_params = [] 
    quantization_params = []
    for n, m in model.named_modules():
        if isinstance(m, SomeFakeQuantize):
            quantization_params.append(m.parameters())
        else:
            normal_params.append(m.parameters())


Then get them into optimizer, e.g.:

.. code-block:: python 
    :linenos:

    # normal_lr 
    # quant_lr 
    # default_lr
    opt = optim.SGD(
        [
            {'params': quantization_params, 'lr': quant_lr},
            {'params': normal_params, 'lr': normal_lr},
        ], lr=default_lr
    )

So, as shown above, a QAT model gets a fake quantization module and its related quantization parameters for subsequent quantization aware training. To learn more about how to add a fake quantize, see also :ref:`add_a_backend_to_mqbench`.
