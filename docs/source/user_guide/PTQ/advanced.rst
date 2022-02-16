Advanced PTQ
========
This part, we introduce some advanced post-training quantization methods including AdaRound, BRECQ and QDrop.
Fair experimental comparisons can be found in Benchmark.

**Adaround**

`AdaRound <https://arxiv.org/pdf/2004.10568.pdf>`_ aims to find the global optimal strategy of rounding the quantized values. In common sense, rounding-to-nearest is optimal for each individual value, but through threoretical analysis on the quantization loss, it's not the case for the entire network or the whole layer. The second order term in the difference contains cross term of the round error, illustrated in a layer of two weights:

.. raw:: latex html

           \[ E[ L(x,y,\mathbf{w}) - L(x,y,\mathbf{w}+\Delta \mathbf{w}) ] \approx \Delta \mathbf{w}^T g^{(\mathbf{w})} + \frac12 \Delta \mathbf{w}^T H^{(\mathbf{w})} \Delta \mathbf{w} \approx \Delta \mathbf{w}_1^2 + \Delta \mathbf{w}_2^2 + \Delta \mathbf{w}_1 \Delta \mathbf{w}_2 \]

Hence, it's benificial to learn a rounding mask for each layer. One well-designed object function is given by the authors:

.. raw:: latex html

           \[ \mathop{\arg\min}_{\mathbf{V}}\ \ || Wx-\tilde{W}x ||_F^2 + \lambda f_{reg}(\mathbf{V}), \]
           \[ \tilde{W}=s \cdot clip\left( \left\lfloor\dfrac{W}{s}\right\rfloor+h(\mathbf{V}), n, p \right) \]

where :math:`h(\mathbf{V}_{i,j})=clip(\sigma(\mathbf{V}_{i,j})(\zeta-\gamma)+\gamma, 0, 1)`, and :math:`f_{reg}(\mathbf{V})=\mathop{\sum}_{i,j}{1-|2h(\mathbf{V}_{i,j})-1|^\beta}`. By annealing on :math:`\beta`, the rounding mask can adapt freely in initial phase and converge to 0 or 1 in later phase. 

.. code-block:: python
    :linenos:

    import torchvision.models as models
    from mqbench.convert_deploy import convert_deploy
    from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization
    from mqbench.adaround import adaround

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)

    # then, we will trace the original model using torch.fx and \
    # insert fake quantize nodes according to different hardware backends (e.g. TensorRT).
    model = prepare_by_platform(model, BackendType.Tensorrt)

    # before training, we recommend to enable observers for calibration in several batches, and then enable quantization.
    model.eval()
    enable_calibration(model)
    calibration_flag = True

    # set adaround config
    adaround_config_dict = {
        adaround: True,
        warm_up: 0.2,
        weight: 0.01,
        max_count: 10000,
        b_range: [20, 2],
        keep_gpu: True,
        round_mode: learned_hard_sigmoid}

    # adaround loop
    stacked_tensor = []
    # add calibration data to stack
    for i, batch_data in enumerate(data):
        if i == cali_batchsize:
            break
        stacked_tensor.append(batch_data)
    # start calibration
    enable_quantization(model)
    model = adaround(model, stacked_tensor, adaround_config_dict)

    # do evaluation
    ...

    # deploy model, remove fake quantize nodes and dump quantization params like clip ranges.
    convert_deploy(model.eval(), BackendType.Tensorrt, input_shape_dict={'data': [10, 3, 224, 224]})


**BRECQ**

Unlike AdaRound, which learn to reconstruct the output and tune the weight layer by layer,
BRECQ discusses different granularity of output reconstruction including layer, block, stage and net.
Combined with experimental results and theoretical analysis, BRECQ recommends to learn weight rounding block by block,
where a block is viewed as collection of layers.

Here, we obey the following rules to determine a block:

    1. A layer is a Conv or Linear module, BN and ReLU are attached to that layer. 

    2. Residual connection should be in the block, such as BasicBlock in ResNet.

    3. If there is no residual connection, singles layers should be combined unless there are 3 single layers or next layer meets condition 2.

**QDrop**

Based on BRECQ, QDrop first compares different orders of optimization procedure (weight and activation) and concludes that 
first weight then activation behaves poorly especially at ultra-low bit. It recommends to let the weight face activation quantization
such as learn the step size of activation and weight rounding together. However, it also points out that there are better ways to do
activation quantization to find a good calibrated weight. Finally, they replace the activation quantization value by FP32 one randomly at netron level
during reconstruction. And they use the probability 0.5 to drop activation quantization.