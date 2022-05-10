Quick Start -- Deploy Just in 4 Lines
================================================

This page is for engineers **who want to deploy models to the production environment using MQBench**, 
if you want to know how to do research with MQBench, check :doc:`quick_start_academic`.

MQBench is a benchmark and framework for evaluating the quantization algorithms under real-world hardware deployments. 
By using MQBench backend presets, you can do **hardware alignment** easily, which means **what you get from MQBench is what you put on your hardware**.

Before learning the internal of MQBench, we provide a simple tutorial to help you to start up your business quickly.
MQBench is designed easy-to-use, for example, you can deploy your FP32 pre-trained model **JUST by inserting 4 lines** of code. 
We provide step-by-step instructions and detailed comments below to help you finish deploying the **PyTorch ResNet18** model to **TensorRT** Backend.

Before starting, you should install MQBench first. Now we start the tour.

**1**. **To begin with, let's import MQBench and prepare FP32 model.**

.. code-block:: python

    import torchvision.models as models                           # for example model
    from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
    from mqbench.prepare_by_platform import BackendType           # contain various Backend, like TensorRT, NNIE, etc.
    from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
    from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8
    from mqbench.convert_deploy import convert_deploy             # remove quant nodes for deploy

    model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
    model.eval()

**2**. **Choose your backend.**

.. code-block:: python

    # backend options
    backend = BackendType.Tensorrt
    # backend = BackendType.SNPE
    # backend = BackendType.PPLW8A16
    # backend = BackendType.NNIE
    # backend = BackendType.Vitis
    # backend = BackendType.ONNX_QNN
    # backend = BackendType.PPLCUDA
    # backend = BackendType.OPENVINO
    # backend = BackendType.Tengine_u8
    # backend = BackendType.Tensorrt_NLP

**3**. **Prepares to quantize the model.**

.. code-block:: python

    model = prepare_by_platform(model, backend)                   #! line 1. trace model and add quant nodes for model on Tensorrt Backend

    # calibration loop
    enable_calibration(model)                                     #! line 2. turn on calibration, ready for gathering data
    for i, batch in enumerate(data):
        # do forward procedures
        ...

    enable_quantization(model)                                    #! line 3. turn on actually quantization, ready for simulating Backend inference

    # evaluation loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

**4**. **Export quantized model.**

.. code-block:: python

    # define dummy data for model export.
    input_shape={'data': [10, 3, 224, 224]}
    convert_deploy(model, backend, input_shape)                   #! line 4. remove quant nodes, ready for deploying to real-world hardware

If you want to know more about deploying to a customize backend, check :doc:`../user_guide/internal/learn_config` and :doc:`../user_guide/howtodeploy`

**Now you can use exported files to test on real hardware using TensorRT as Backend, congratulations!**

Now you can follow our advanced :doc:`user guide <../developer_guide/index>` and :doc:`developer guide <../user_guide/index>` to know more about MQBench.
