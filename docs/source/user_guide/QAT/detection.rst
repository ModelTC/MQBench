Object detection with MQBench
================================

This part, we introduce how to quantize an object detection model using MQBench.

Getting Started
---------------

**1**. **Clone the repositories.**

.. code-block:: shell

    git clone https://github.com/ModelTC/MQBench.git
    git clone https://github.com/ModelTC/EOD.git

**2**. **Quantization aware training.**

.. code-block:: shell

    # Prepare your float pretrained model.
    cd eod/scripts
    # Follow the prompts to set config in train_quant.sh.
    sh train_qat.sh


**We have several examples of qat config in EOD repository:**

For retinanet-tensorrt:
 - float pretrained config file: retinanet-r18-improve.yaml
 - qat config file: retinanet-r18-improve_quant_trt_qat.yaml

For yolox-tensorrt:
 - float pretrained config file: yolox_s_ret_a1_comloc.yaml
 - qat config file: yolox_s_ret_a1_comloc_quant_trt_qat.yaml

For yolox-vitis:
 - float pretrained config file: yolox_fpga.yaml
 - qat config file: yolox_fpga_quant_vitis_qat.yaml

**Something import in config file:**

 - deploy_backend: Choose your deploy backend supported in MQBench.
 - ptq_only: If True, only ptq will be executed. If False, qat will be executed after ptq calibration.
 - extra_qconfig_dict: Choose your quantization config supported in MQBench.
 - leaf_module: Prevent torch.fx tool entering the module.
 - extra_quantizer_dict: Add some qat modules.
 - resume_model: The path to your float pretrained model.
 - tocaffe_friendly: It is recommended to set it to true, which will affect the output onnx model.

**3**. **Resume training during qat.**

.. code-block:: shell

    cd eod/scripts
    # just set resume_model in config file to your model, we will do all the rest.
    sh train_qat.sh


**4**. **Evaluate your quantized model.**

.. code-block:: shell

    cd eod/scripts
    # set resume_model in config file to your model
    # add -e to train_qat.sh
    sh train_qat.sh

**5**. **Deploy.**

.. code-block:: shell

    cd eod/scripts
    # Follow the prompts to set config in quant_deploy.sh.
    sh qat_deploy.sh

Introduction of EOD-MQBench Project
----------------------------------------

Code related to quantization is in eod/tasks/quant.

When you set the runner type to quant in config file, QuantRunner will be executed in eod/tasks/quant/runner/quant_runner.py.

1. Firstly, build your float model in self.build_model().
2. Load your float pretrained model/quantized model in self.load_ckpt().
3. Use torch.fx to trace your model in self.quantize_model().
4. Set your optimization and lr scheduler in self.build_trainer().
5. Ptq and eval in self.calibrate()
6. Train in self.train()

**Something important:**

 - Your model should be split into network and post-processing. Fx should only trace the network.
 - Quantized model should be saved with the key of qat, as shown in self.save(). This will be used in self.resume_model_from_fp() and self.resume_model_from_quant().
 - We disable the ema in qat. If your ckpt has ema state, we will load ema state into model, as shown in self.load_ckpt().
 - Be careful when your quantized model has extra learnable parameters. You can check it in optimizer, such as eod/tasks/det/plugins/yolov5/utils/optimizer_helper.py. Lsq has been checked.
