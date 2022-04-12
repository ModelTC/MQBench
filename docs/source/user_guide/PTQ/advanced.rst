Advanced PTQ
============

Code Snippets
^^^^^^^^^^^^^

You can follow this snippet to start your mission with MQBench! You can find config details in `here <https://github.com/ModelTC/MQBench/tree/main/application/imagenet_example/PTQ/configs>`_, and you can find algorithm details in :doc:`../algorithm/advanced_ptq`.


.. code-block:: python
    :linenos:

    import torchvision.models as models
    from mqbench.convert_deploy import convert_deploy
    from mqbench.prepare_by_platform import prepare_qat_fx_by_platform, BackendType
    from mqbench.utils.state import enable_calibration, enable_quantization
    from mqbench.advanced_ptq import ptq_reconstruction

    # first, initialize the FP32 model with pretrained parameters.
    model = models.__dict__["resnet18"](pretrained=True)

    # then, we will trace the original model using torch.fx and \
    # insert fake quantize nodes according to different hardware backends (e.g. TensorRT).
    model = prepare_by_platform(model, BackendType.Tensorrt)

    # before training, we recommend to enable observers for calibration in several batches, and then enable quantization.
    model.eval()
    enable_calibration(model)
    calibration_flag = True

    # set config
    config_dict = {
        pattern: 'block',
        warm_up: 0.2,
        weight: 0.01,
        max_count: 10000,
        b_range: [20, 2],
        keep_gpu: True,
        round_mode: learned_hard_sigmoid,
        prob: 1.0
        }

    # ptq_reconstruction loop
    stacked_tensor = []
    # add calibration data to stack
    for i, batch_data in enumerate(data):
        if i == cali_batchsize:
            break
        stacked_tensor.append(batch_data)
    # start calibration
    enable_quantization(model)
    model = ptq_reconstruction(model, stacked_tensor, config_dict)

    # do evaluation
    ...

    # deploy model, remove fake quantize nodes, and dump quantization params like clip ranges.
    convert_deploy(model.eval(), BackendType.Tensorrt, input_shape_dict={'data': [10, 3, 224, 224]})

MQBench examples
^^^^^^^^^^^^^^^^^

We follow the `PyTorch official example <https://github.com/pytorch/examples/tree/master/imagenet/>`_ to build the example of Model Quantization Benchmark for ImageNet classification task, you can run advanced ptq easily.

1. Clone and install MQBench;
2. Prepare the ImageNet dataset from `the official website <http://www.image-net.org/>`_ and move validation images to labeled subfolders, using the following `shell script <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>`_;
3. Download pre-trained models from our `release <https://github.com/ModelTC/MQBench/releases/tag/pre-trained>`_;
4. Check out `/path-of-MQBench/application/imagenet_example/PTQ/configs` and find yaml file you want to reproduce;
5. Replace `/path-of-pretained` and `/path-of-imagenet` in yaml file;
6. Change directory, `cd /path-of-MQBench/application/imagenet_example/PTQ/ptq`;
7. Exec `python ptq.py -\-config /path-of-config.yaml`.
