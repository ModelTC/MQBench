Quantization Algorithm
===========================

.. _LSQ: https://arxiv.org/abs/1902.08153
.. _LSQ plus: https://arxiv.org/abs/2004.09576
.. _DSQ: https://arxiv.org/abs/1908.05033
.. _PACT: https://arxiv.org/abs/1805.06085
.. _APoT: https://arxiv.org/abs/1909.13144
.. _opensource codes: https://github.com/yhhhli/APoT_Quantization
.. _weight standardization: https://github.com/joe-siyuan-qiao/WeightStandardization
.. _QIL: https://arxiv.org/abs/1808.05779
.. _AdaRound: https://arxiv.org/abs/2004.10568


Post-training Quantization v.s. Quantization-aware Training
-----------------------------------------------------------------------

1. Post Training Quantization (PTQ):

   Quantize a pre-trained network with limited data and computation resources, including activation range estimation, bn statistics update and other tuning techniques. 

2. Quantization Aware Training (QAT):

   End-to-end Finetuning a pre-trained full-precision model, this requires all training data and huge computation resource. 

QAT Algorithms
---------------------------------

**Learned Step Size Quantization**:

`LSQ`_ leverages the Straight-Through Estimator (i.e. directly pass the gradient in round operation) to learn the quantization scale for each layer. 
Please refer to the original paper for detailed derivation of the scale gradient. 
For initialization, we use the method proposed in original paper: the scale is determined by :math:`s= \frac{2||\mathbf{w}||_1}{\sqrt{N_{max}}}`. For symmetric quantization, the zero point is initialized to 0, and kept fixed. For asymmetric quantization, zero point is initialized to :math:`N_{min}` if the activation is non-negative. Inspired by `LSQ plus`_, the zero point can also be updated through backpropagation with the help of STE. Therefore we make it learnable in asymmetric quantization. 
LSQ uses gradient scale to stabilize the scale learning. The gradient scale is determined by :math:`\frac{1}{\sqrt{MN_{max}}}` where :math:`M` is the number of elements in that tensor. We extend this gradient scale to per-channel weight learning, where the :math:`M` is the number of weights in each filter. 


**Differentiable Soft Quantization**:

`DSQ`_ uses the hyperbolic tangent function to approximate the conventionally adopted STE. In our implementation, we use :math:`\alpha=0.4` (for definition please refer to the original paper) which controls the shape and smoothness of the :math:`\mathrm{tanh}` function. For weight quantization, we use the min-max range as

.. raw:: latex html

           \[Clip_{min} = \mu(\mathbf{w}) - 2.6\sigma(\mathbf{w}) \]
           \[Clip_{max} = \mu(\mathbf{w}) + 2.6\sigma(\mathbf{w}) \]


where :math:`\mu(\cdot)` and :math:`\sigma(\cdot)` compute the mean and standard deviation of the tensor. Then, the scale is determined by :math:`s=\frac{\max(-Clip_{min}, Clip_{max})}{N_{max}-N_{min}}` for symmetric quantization, and :math:`\frac{Clip_{max}-Clip_{min}}{N_{max}-N_{min}}` for asymmetric quantization. The zero point is set to 0 for symmetric and :math:`N_{min}-\lfloor \frac{Clip_{min}}{s}\rceil` for asymmetric quantization. For activation, we use the BatchMinMax as the clipping range, i.e. the averaged min-max range across the batch dimension. This is further updated with exponential moving average across different batches with momentum 0.9, similar to Batch Normalization.

**Parameterized Clipping Activation**:

`PACT`_ is introduced to quantized activation by learning the clipping threshold through STE. Its activation is clipped by a parameter :math:`\alpha` first. Then, the clipped activation is quantized and re-quantized. Although PACT and LSQ both learns the scale, they have three differences. First, the clipping range in PACT is handcrafted initialized to 6 while LSQ initialization is based on the tensor :math:`L1` norm. Second, PACT has no gradient in the range of clipping. While LSQ can compute the gradient. Third, PACT does not scale the gradient of :math:`\alpha`, while LSQ does. 
Note that PACT only has non-negative, unsigned quantization in the first. To extend it to our hardware settings, we clip the activation to :math:`(-\alpha, \alpha)` in symmetric case and :math:`(\beta, \alpha)` for asymmetric case, (where :math:`\beta` is initialized to :math:`-6`).
For weight quantization of PACT, it is the same with DoReFa-Net.

**DoReFa-Net**:

DoReFa-Net simply clips the activation to :math:`[0, 1]` and then quantizes it. This is based on the intuition that most activation will fall into this range in old network architectures, e.g. AlexNet and ResNet. In hardware settings, we modify the activation range to :math:`[-1, 1]` for both symmetric and asymmetric quantization. As for weight quantization, it can be described as: 

.. raw:: latex html

           \[\tilde{\mathbf{w}} = \mathrm{tanh}(\mathbf{w}) \frac{1}{\max(|\mathrm{tanh}(\mathbf{w})|)} \]
           \[\hat{\mathbf{w}} = \mathrm{dequantize}(\mathrm{quantize(\tilde{\mathbf{w}})}) \]

where the first step is a non-linear transformation and the second step is the quantization and the de-quantization. The scale is simply calculated by :math:`\frac{2}{N_{max}-N_{min}}` for symmetric quantization and :math:`\frac{\max(\tilde{\mathbf{w}}) - \min(\tilde{\mathbf{w}})}{N_{max}-N_{min}}` for asymmetric quantization. 


**Additive Powers-of-Two Quantization**:

`APoT`_ quantization uses multiple PoT's (Powers-of-Two)  combination to composes a set of non-uniform quantization levels. Since the quantization are non-uniform in most cases (except the case of 2-bit the APoT becomes uniform quantization), we do not benchmark it on real hardware. Additionally, APoT introduces weight normalization (similar to `weight standardization`_ technique) to smooth the learning process of clipping range in weight. However, it is unclear how to incoporate this technique with BN folding. 
Therefore, we only reproduce it in our academic setting. The implementation are based on the `opensource codes`_. 



**Quantization Interval Learning**:

`QIL`_ composes of two unit to quantization: (1) the first one is called transformer, which transform the weights or activation to :math:`[-1, 1]` (:math:`[0, 1]` as for non-negative activation). 
This transformer also has two functionalities: pruning and non-linearity. 
(2) The second one is called quantizer, given by

.. raw:: latex html

           \[ \tilde{\mathbf{w}} = \mathrm{clip}\left((\alpha |\mathbf{w}| + \beta)^{\gamma}, 0, 1\right) * \mathrm{sign}(\mathbf{w})\]
           \[    \hat{\mathbf{w}} = \mathrm{dequantize}(\mathrm{quantize(\tilde{\mathbf{w}})}),  \]

where :math:`\alpha = \frac{1}{2*D}` and :math:`\beta=-\frac{C}{2D}+\frac{1}{2}`. This transformation maps the weight from :math:`[C-D, C+D]` to :math:`[0, 1]` and :math:`[-C-D, -C+D]` to :math:`[-1, 0]`. As a result, the weights between :math:`[-C+D, C-D]` are pruned. The non-linearity of the transformation function is introduced by $\gamma$. This parameter can control the linearity and thus control the quantization interval. However, we find this technique is extremely unstable. In our experimental reproduction, learning $\gamma$ will not converge. In the original paper, the gradient scale of :math:`C` and :math:`D` is set to 0.01. We find this gradient scale also leads to frequent crashes. Thus we use the gradient scale introduced in LSQ, i.e. :math:`\frac{1}{\sqrt{MN_{max}}}`.


PTQ Algorithms
------------------------------

**AdaRound**:

`AdaRound`_ aims to find the global optimal strategy of rounding the quantized values. In common sense, rounding-to-nearest is optimal for each individual value, but through threoretical analysis on the quantization loss, it's not the case for the entire network or the whole layer. The second order term in the difference contains cross term of the round error, illustrated in a layer of two weights:

.. raw:: latex html

           \[ E[ L(x,y,\mathbf{w}) - L(x,y,\mathbf{w}+\Delta \mathbf{w}) ] \approx \Delta \mathbf{w}^T g^{(\mathbf{w})} + \frac12 \Delta \mathbf{w}^T H^{(\mathbf{w})} \Delta \mathbf{w} \approx \Delta \mathbf{w}_1^2 + \Delta \mathbf{w}_2^2 + \Delta \mathbf{w}_1 \Delta \mathbf{w}_2 \]

Hence, it's benificial to learn a rounding mask for each layer. One well-designed object function is given by the authors:

.. raw:: latex html

           \[ \mathop{\arg\min}_{\mathbf{V}}\ \ || Wx-\tilde{W}x ||_F^2 + \lambda f_{reg}(\mathbf{V}), \]
           \[ \tilde{W}=s \cdot clip\left( \left\lfloor\dfrac{W}{s}\right\rfloor+h(\mathbf{V}), n, p \right) \]

where :math:`h(\mathbf{V}_{i,j})=clip(\sigma(\mathbf{V}_{i,j})(\zeta-\gamma)+\gamma, 0, 1)`, and :math:`f_{reg}(\mathbf{V})=\mathop{\sum}_{i,j}{1-|2h(\mathbf{V}_{i,j})-1|^\beta}`. By annealing on :math:`\beta`, the rounding mask can adapt freely in initial phase and converge to 0 or 1 in later phase. 

