SNPE
=============
Example of QAT and deployment on SNPE.

**Requirements**:

- Install SNPE SDK from `QualComm <https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html>`_ (Suggest Ubuntu 18.04)

**QAT**:

- Follow the QAT procedures to get a model checkpoint, and suggest learning rate as 5e-5 with cosine scheduler and Adam optimizer for tens of epochs.

**Deployment**:

- Convert PyTorch checkpoint to `snpe_deploy.onnx` and dump clip ranges to `snpe_clip_ranges.json`::

    from mqbench.convert_deploy import convert_deploy
    input_dict = {'x': [1, 3, 224, 224]}
    convert_deploy(solver.model.module, BackendType.SNPE, input_dict)

- Convert `.onnx` file to `.dlc` format (supported by SNPE)::

    snpe-onnx-to-dlc --input_network ./snpe_deploy.onnx --output_path ./snpe_deploy.dlc --quantization_overrides ./snpe_clip_ranges.json

    - Note that, the `.json` file contains activation ranges for quantization, but it's required here although the model hasn't been quantized now.

- Quantize the model with parameters overridden::

    snpe-dlc-quantize --input_dlc ./snpe_deploy.dlc --input_list ./data.txt --override_params  --bias_bitwidth 32

    - The `data.txt` records paths to image data for calibration (not important since we will override parameters) which will be loaded by `numpy.fromfile(dtype=np.float32)` and have shape of `(224, 224, 3)`. And this file is required for test.

    - Now we get the final model `snpe_deploy_quantized.dlc`

**Results**:

The test is done by SNPE SDK tools, with the quantized model and a text file recording paths to test data in shape of (224, 224, 3)::

    snpe-net-run --container ./snpe_deploy_quantized.dlc --input_list ./test_data.txt

The results on several tested models:

+-------------------+--------------------------------+------------------------------------------------------------------------------------------------------------------+
|   Model           |       accuracy\@fp32           |                                           accuracy\@int8                                                         |
|                   |                                +-------------------------------------------------------+----------------------------------------------------------+
|                   |                                |                      MQBench QAT                      |                            SNPE                          |
+===================+================================+=======================================================+==========================================================+
|  **ResNet18**     |    70.65%                      |                      70.75%                           |                      70.74%                              |
+-------------------+--------------------------------+-------------------------------------------------------+----------------------------------------------------------+
|  **ResNet50**     |    77.94%                      |                      77.75%                           |                      77.92%                              |
+-------------------+--------------------------------+-------------------------------------------------------+----------------------------------------------------------+
|  **MobileNet_v2** |    72.67%                      |                      72.31%                           |                      72.65%                              |
+-------------------+--------------------------------+-------------------------------------------------------+----------------------------------------------------------+
