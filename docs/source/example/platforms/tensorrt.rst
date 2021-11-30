TensorRT
================
Example of QAT and deployment on TensorRT.

**Requirements**:

- Install TensorRT=7.2.1.6 from `NVIDIA <https://developer.nvidia.com/tensorrt/>`_

**QAT**:

- Training hyper-parameters:

- batch size = 128
- epochs = 1
- learning rate  = 1e-4 (for ResNet series) / 1e-5 (for MobileNet series)
- weight decay = 1e-4 (for ResNet series) / 0 (for MobileNet series)
- optimizer: SGD (for ResNet series) / Adam (for MobileNet series)
- others like momentum are kept as default.

- [model_name] = ResNet18 / ResNet50 / MobileNet_v2 / ... ::

      git clone https://github.com/TheGreatCold/MQBench.git
      cd application/imagenet_example
      python main.py -a [model_name] --epochs 1 --lr 1e-4 --b 128 --pretrained


**Deployment**:

We provide the example to deploy the quantized model to TensorRT.

- First export the quantized model to ONNX [tensorrt_deploy_model.onnx] and dump the clip ranges [tensorrt_clip_ranges.json] for activations. ::

      python main.py -a [model_name] --resume [model_save_path]

- Second build the TensorRT INT8 engine and evaluate, please make sure [dataset_path] contains subfolder [val]. ::

      python onnx2trt.py --onnx [tensorrt_deploy_model.onnx] --trt [model_name.trt] --clip [tensorrt_clip_ranges.json] --data [dataset_path] --evaluate

- If you don’t pass in external clip ranges [tensorrt_clip_ranges.json], TenosrRT will do calibration using default algorithm IInt8EntropyCalibrator2 with 100 images. So, please make sure [dataset_path] contains subfolder [cali]. ::

      python onnx2trt.py --onnx [tensorrt_deploy_model.onnx] --trt [model_name.trt] --data [dataset_path] --evaluate

**Results**:


+-------------------+--------------------------------+------------------------------------------------------------------------------------------------------------------+
|   Model           |       accuracy\@fp32           |                                           accuracy\@int8                                                         |
|                   |                                +----------------------------------------+---------------------------------+---------------------------------------+
|                   |                                |     TensoRT Calibration                |        MQBench QAT              |       TensorRT SetRange               |
+===================+================================+========================================+=================================+=======================================+
|  **ResNet18**     |    Acc\@1 69.758 Acc\@5 89.078 |   Acc\@1 69.612 Acc\@5 88.980          |    Acc\@1 69.912 Acc\@5 89.150  |    Acc\@1 69.904 Acc\@5 89.182        |
+-------------------+--------------------------------+----------------------------------------+---------------------------------+---------------------------------------+
|  **ResNet50**     |    Acc\@1 76.130 Acc\@5 92.862 |   Acc\@1 76.074 Acc\@5 92.892          |    Acc\@1 76.114 Acc\@5 92.946  |    Acc\@1 76.320 Acc\@5 93.006        |
+-------------------+--------------------------------+----------------------------------------+---------------------------------+---------------------------------------+
|  **MobileNet_v2** |    Acc\@1 71.878 Acc\@5 90.286 |   Acc\@1 70.700 Acc\@5 89.708          |    Acc\@1 71.158 Acc\@5 89.990  |    Acc\@1 71.102 Acc\@5 89.932        |
+-------------------+--------------------------------+----------------------------------------+---------------------------------+---------------------------------------+
