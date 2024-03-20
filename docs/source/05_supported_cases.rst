Supported Cases
=====================

We have prepared ready-to-execute quantization scripts in Sophgo-MQ. In this section, we will demonstrate how to use scripts for quantizing different models.



CV model PTQ
-------------------------------

.. code-block:: bash

  CUDA_VISIBLE_DEVICES=0 python application/imagenet_example/PTQ/ptq/ptq_main.py \
      --arch=resnet18 \
      --batch-size=64 \
      --cali-batch-num=16 \
      --data_path=/home/data/imagenet \
      --chip=SG2260 \
      --quantmode=weight_activation \
      --seed=1005 \
      --pretrained \
      --quantize_type=naive_ptq \
      --deploy \
      --output_path=./


CV model QAT
-------------------------------

.. code-block:: bash

  CUDA_VISIBLE_DEVICES=0 python application/imagenet_example/main.py \
      --arch=resnet18 \
      --batch-size=128 \
      --lr=1e-4 \
      --epochs=1 \
      --optim=sgd \
      --cuda=0 \
      --pretrained \
      --evaluate \
      --train_data=/home/data/imagenet \
      --val_data=/home/data/imagenet \
      --chip=SG2260 \
      --quantmode=weight_activation \
      --deploy_batch_size=10 \
      --pre_eval_and_export \
      --export_onnx_before_training \
      --output_path=./

Sophgo-mq also supports QAT quantization for Yolov5.

.. code-block:: bash

  cd ./application/yolov5_example
  export PYTHONPATH=../../:$PYTHONPATH
  CUDA_VISIBLE_DEVICES=0 python train.py \
      --cfg=yolov5s.yaml \
      --weights=yolov5s.pt \
      --data=coco.yaml \
      --epochs=5 \
      --output_path=./ \
      --batch-size=8 \
      --quantize \


NLP model PTQ
-------------------------------

Sophgo-mq supports PTQ quantization for the BERT model on the MRPC dataset.

.. code-block:: bash

  cd ./application/nlp_example
  export PYTHONPATH=../../:$PYTHONPATH
  python ptq.py --config config.yaml



NLP model QAT
-------------------------------

Sophgo-mq supports QAT quantization for the BERT model on the SQuAD dataset.

.. code-block:: bash

  cd ./application/nlp_example
  export PYTHONPATH=../../:$PYTHONPATH
  python qat_bertbase_questionanswer.py