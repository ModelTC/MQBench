Get Started
==========================
We follow the `PyTorch official example <https://github.com/pytorch/examples/tree/master/imagenet/>`_ to build the example of Model Quantization Benchmark for ImageNet classification task.
 
Requirements
-------------

- Install PyTorch following `pytorch.org <http://pytorch.org/>`_
- Install dependencies::

    pip install -r requirements.txt

- Download the ImageNet dataset from `the official website <http://www.image-net.org/>`_

  - Then, and move validation images to labeled subfolders, using `the following shell script <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh/>`_

- Install TensorRT=7.2.1.6 from `NVIDIA <https://developer.nvidia.com/tensorrt/>`_

Usage
---------

- **Quantization-Aware Training:**

  - Training hyper-parameters:
    
    - batch size = 128
    - epochs = 1 
    - lr = 1e-4
    - others like weight decay, momentum are kept as default.

  - ResNet18 / ResNet50 / MobileNet_v2::

        python main.py -a [model_name] --epochs 1 --lr 1e-4 --b 128 --seed 99 --pretrained


- **Deployment**
  We provide the example to deploy the quantized model to TensorRT.

  1. First export the quantized model to ONNX [tensorrt_deploy_model.onnx] and dump the clip ranges [tensorrt_clip_ranges.json] for activations.::

        python main.py -a [model_name] --resume [model_save_path]
     

  2. Second build the TensorRT INT8 engine and evaluate, please make sure [dataset_path] contains subfolder [val]::

        python onnx2trt.py --onnx [tensorrt_deploy_model.onnx] --trt [model_name.trt] --clip [tensorrt_clip_ranges.json] --data [dataset_path] --evaluate
    
  3. If you don’t pass in external clip ranges [tensorrt_clip_ranges.json], TenosrRT will do calibration using default algorithm IInt8EntropyCalibrator2 with 100 images. So, please make sure [dataset_path] contains subfolder [cali]::

        python onnx2trt.py --onnx [tensorrt_deploy_model.onnx] --trt [model_name.trt] --data [dataset_path] --evaluate

Results
-----------

+-------------------+--------------------------------+------------------------------------------------------------------------------------------------------------------+
|   Model           |       accuracy\@fp32           |                                           accuracy\@int8                                                         |
|                   |                                +----------------------------------------+---------------------------------+---------------------------------------+
|                   |                                |     TensoRT Calibration                |        MQBench QAT              |       TensorRT SetRange               |  
+===================+================================+========================================+=================================+=======================================+
|  **ResNet18**     |    Acc\@1 69.758 Acc\@5 89.078 |   Acc\@1 69.612 Acc\@5 88.980          |    Acc\@1 69.912 Acc\@5 89.150  |    Acc\@1 69.904 Acc\@5 89.182        |
+-------------------+--------------------------------+----------------------------------------+---------------------------------+---------------------------------------+ 
|  **ResNet50**     |    Acc\@1 76.130 Acc\@5 92.862 |   Acc\@1 76.074 Acc\@5 92.892          |    Acc\@1 76.114 Acc\@5 92.946  |    Acc\@1 76.320 Acc\@5 93.006        | 
+-------------------+--------------------------------+----------------------------------------+---------------------------------+---------------------------------------+
|  **MobileNet_v2** |    Acc\@1 71.878 Acc\@5 90.286 |   Acc\@1 70.700 Acc\@5 89.708          |    Acc\@1 70.826 Acc\@5 89.874  |    Acc\@1 70.724 Acc\@5 89.870        |  
+-------------------+--------------------------------+----------------------------------------+---------------------------------+---------------------------------------+
