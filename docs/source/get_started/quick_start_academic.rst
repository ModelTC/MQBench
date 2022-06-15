Quick Start -- Embrace Best Research Experience
=================================================

This page is for researchers **who want to validate their marvelous quantization idea using MQBench**, if you want to get started with deployment using MQBench, check :doc:`quick_start_deploy`.

MQBench is a benchmark, a framework and a good tool for researchers. MQBench is designed easy-to-use for researchers, for example, you can easily custom Academic Backend by providing an extra config dict to conduct any experiment.
We provide step-by-step instructions and detailed comments below to help you finish deploying the **PyTorch ResNet18** model to a **Custom Academic** Backend.

Before starting, you should have done the MQBench setup in :doc:`setup`. Now we start the tour.

**1**. **To begin with, let's import MQBench and prepare FP32 model.**

.. code-block:: python

    import torchvision.models as models                           # for example model
    from mqbench.prepare_by_platform import prepare_by_platform   # add quant nodes for specific Backend
    from mqbench.prepare_by_platform import BackendType           # contain various Backend, contains Academic.
    from mqbench.utils.state import enable_calibration            # turn on calibration algorithm, determine scale, zero_point, etc.
    from mqbench.utils.state import enable_quantization           # turn on actually quantization, like FP32 -> INT8

    model = models.__dict__["resnet18"](pretrained=True)          # use vision pre-defined model
    model.eval()

**2**. **Then we learn the extra configration to custom Academic Backend.**

You can also learn this section through MQBench `source code <https://github.com/ModelTC/MQBench/blob/main/mqbench/prepare_by_platform.py#L125>`_.
Learn all options through our :doc:`../user_guide/internal/learn_config`

.. code-block:: python

    extra_config = {
        'extra_qconfig_dict': {
            'w_observer': 'MSEObserver',                              # custom weight observer
            'a_observer': 'MSEObserver',                              # custom activation observer
            'w_fakequantize': 'FixedFakeQuantize',                    # custom weight fake quantize function
            'a_fakequantize': 'FixedFakeQuantize',                    # custom activation fake quantize function
            'w_qscheme': {
                'bit': 8,                                             # custom bitwidth for weight,
                'symmetry': False,                                    # custom whether quant is symmetric for weight,
                'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for weight,
                'pot_scale': False,                                   # custom whether scale is power of two for weight.
            },
            'a_qscheme': {
                'bit': 8,                                             # custom bitwidth for activation,
                'symmetry': False,                                    # custom whether quant is symmetric for activation,
                'per_channel': True,                                  # custom whether quant is per-channel or per-tensor for activation,
                'pot_scale': False,                                   # custom whether scale is power of two for activation.
            }
        }
    }

**3**. **The next step prepares to conduct the experiment, take PTQ as the example.**

.. code-block:: python

    model = prepare_by_platform(model,
        BackendType.Academic, extra_config)                       #! 1. trace model and add quant nodes for model on Academic Backend

    enable_calibration(model)                                     #! 2. turn on calibration, ready for gathering data

    # calibration loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

    enable_quantization(model)                                    #! 3. turn on actually quantization, ready for simulating Backend inference

    # evaluation loop
    for i, batch in enumerate(data):
        # do forward procedures
        ...

**You have already known all basics about how to validate your marvelous quantization idea with MQBench, congratulations!**

Now you can follow our advanced :doc:`user guide <../developer_guide/index>` and :doc:`developer guide <../user_guide/index>` to know more about MQBench.
