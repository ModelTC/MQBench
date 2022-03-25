Image Classification Benchmark
==============================

How to reproduce our results?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Generally, we follow the `PyTorch official example <https://github.com/pytorch/examples/tree/master/imagenet/>`_ to build the example of Model Quantization Benchmark for ImageNet classification task.

1. Clone and install MQBench;
2. Prepare the ImageNet dataset from `the official website <http://www.image-net.org/>`_ and move validation images to labeled subfolders, using the following `shell script <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>`_;
3. Download pre-trained models from our `release <https://github.com/ModelTC/MQBench/releases/tag/pre-trained>`_;
4. Check out `/path-of-MQBench/application/imagenet_example/PTQ/configs` and find yaml file you want to reproduce;
5. Replace `/path-of-pretained` and `/path-of-imagenet` in yaml file;
6. Change directory, `cd /path-of-MQBench/application/imagenet_example/PTQ/ptq`;
7. Exec `python ptq.py -\-config /path-of-config.yaml`.


.. _imagenet-ptq-benchmark:

Post-training Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Backend: Academic

+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| W_calibration | A_calibration | Backend  | wbit | abit | resnet18 | resnet50 | mobilenetv2_1.0 | regnetx600m | regnetx800m |
+===============+===============+==========+======+======+==========+==========+=================+=============+=============+
| None          | None          | Academic | 32   | 32   | 71.06    | 76.63    | 72.68           | 73.60       | 74.83       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| MinMax        | EMAMinMax     | Academic | 8    | 8    | 70.93    | 76.49    | 72.02           | 73.48       | 74.85       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| MinMax        | EMAQuantile   | Academic | 8    | 8    | 70.89    | 76.49    | 72.02           | 73.51       | 74.82       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| MSE           | EMAMSE        | Academic | 8    | 8    | 70.88    | 76.58    | 72.02           | 73.61       | 74.83       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| MinMax        | EMAMinMax     | Academic | 4    | 8    | 52.25    | 70.34    | 27.61           | 60.37       | 57.25       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| MinMax        | EMAQuantile   | Academic | 4    | 8    | 52.20    | 70.34    | 27.61           | 60.26       | 57.38       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| MSE           | EMAMSE        | Academic | 4    | 8    | 54.96    | 69.44    | 35.71           | 62.30       | 57.93       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| AdaRound      | EMAMSE        | Academic | 4    | 8    | 70.35    | 76.87    | 71.82           | 72.32       | 73.58       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+


+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| W_calibration | A_calibration | Backend  | wbit | abit | resnet18 | resnet50 | mobilenetv2_1.0 | regnetx600m | regnetx800m |
+===============+===============+==========+======+======+==========+==========+=================+=============+=============+
| None          | None          | Academic | 32   | 32   | 71.06    | 76.63    | 72.68           | 73.60       | 74.83       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| AdaRound      | EMAMSE        | Academic | 4    | 4    | 68.67    | 73.79    | 65.74           | 70.24       | 71.54       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| BRECQ         | EMAMSE        | Academic | 4    | 4    | 68.52    | 74.66    | 67.23           | 70.30       | 72.04       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| QDrop         | EMAMSE        | Academic | 4    | 4    | 68.84    | 74.98    | 68.13           | 70.85       | 72.62       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| AdaRound      | EMAMSE        | Academic | 2    | 4    | 62.31    | 63.65    | 40.60           | 57.14       | 58.33       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| BRECQ         | EMAMSE        | Academic | 2    | 4    | 63.56    | 68.39    | 52.29           | 62.36       | 64.53       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| QDrop         | EMAMSE        | Academic | 2    | 4    | 64.49    | 69.15    | 53.47           | 63.51       | 65.84       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| AdaRound      | EMAMSE        | Academic | 3    | 3    | 64.18    | 64.76    | 33.61           | 59.57       | 61.45       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| BRECQ         | EMAMSE        | Academic | 3    | 3    | 64.24    | 70.17    | 51.86           | 62.83       | 65.49       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+
| QDrop         | EMAMSE        | Academic | 3    | 3    | 65.42    | 71.07    | 56.05           | 64.78       | 67.45       |
+---------------+---------------+----------+------+------+----------+----------+-----------------+-------------+-------------+

.. note::
  Although AdaRound and BRECQ first learn the weight rounding with FP32 activation then determine the quantization parameters,
  we find let weight face activation quantization behaves better,
  extremely for ultra-low bit as proposed in QDrop.
  Therefore, here we take the same training strategy as QDrop for fair comparisons among these three methods.
  Hyperparameters are also kept the same except that AdaRound uses 10000 iters to do layer reconstruction
  and BRECQ, QDrop use 20000 iters for block reconstruction.

.. note::
  About block partition in MobileNetV2 in BRECQ and QDrop, we achieve it somewhere different with their original paper
  for the sake of more general and automatic partition way.

