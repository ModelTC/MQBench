Vitis
========

Introduction
^^^^^^^^^^^^

'Xilinx Vitis <https://github.com/Xilinx/Vitis-AI/>`_ is a platform for high-performance deep learning inference on Xilinx FPGA device.

.. _Vitis Quantization Scheme:

**Quantization Scheme**

8bit per-tensor power-of-two symmetric linear quantization.

.. math::

    \begin{equation}
        q = \mathtt{clamp}(\lfloor x / s \rceil, lb, ub) * s
    \end{equation}


where :math:`s` is power-of-two scaling factor to quantize a number from floating range to integer range, :math:`lb` and :math:`ub` are bounds of integer range.
For weights and activations, [lb, ub] = [-128, 127].

Deploy on Vitis
^^^^^^^^^^^^^^^^^^

**Requirements**:

- Build Docker from /docker

**Deployment**:

We provide the example to deploy the quantized EOD model to Vitis.

- First modify the configuration file, add quantization, taks yolox_tiny as an example, save new configuration file as "yolox_tiny_quant.yaml".
    
    .. code-block:: shell
        :linenos:

        quant:
  	  deploy_backend: vitis
          cali_batch_size: 50

- Second change optimizer and lr_scheduler.
    
    .. code-block:: shell
        :linenos:
   
    	trainer: 
   	  max_epoch: &max_epoch 5
  	  save_freq: 1
  	  test_freq: 1
  	  optimizer:             
            register_type: qat_weights
            type: Adam
            kwargs:
              lr: 0.00000015625
              weight_decay: 0.000
          lr_scheduler:
            type: MultiStepLR
              kwargs:
                milestones: [1,2]
                gamma: 0.1

- Third quantize model.

    .. code-block:: shell
        :linenos:

	python -m eod train --config configs/det/yolox/yolox_tiny_quant.yaml --nm 1 --ng 1 --launch pytorch

- Fourth use function deploy() in ./eod/runner/quantexport eport deployed model xmodel[mqbench_qmodel.xmodel].

    .. code-block:: shell
        :linenos:

        from mqbench.convert_deploy import convert_deploy
        deploy_backend = self.config['quant']['deploy_backend']
        dummy_input = self.get_batch('train')
        self.model.eval()
        convert_deploy(self.model, self.backend_type[deploy_backend], dummy_input={'image': dummy_input['image']})
