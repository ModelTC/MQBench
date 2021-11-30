Preparations
===================================
Generally, we follow the `PyTorch official example <https://github.com/pytorch/examples/tree/master/imagenet/>`_ to build the example of Model Quantization Benchmark for ImageNet classification task.


- Install PyTorch following `pytorch.org <http://pytorch.org/>`_
- Install dependencies ::

    pip install -r requirements.txt

- Specific requirements for hardware platforms will be introduced later

- Download the ImageNet dataset from `the official website <http://www.image-net.org/>`_

  - Then, and move validation images to labeled subfolders, using `the following shell script <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh/>`_

  - Or process other datasets in the similar way.

- Full precision pretrained models are preferred, but sometimes it's possible to do QAT from scratch.


