
# # export onnx
# python main.py -a resnet18 --resume resnet18.69978.pth --gpu 2 --deploy --backend snpe \
#     --train_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC \
#     --val_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC

# # eval qat model
# python main.py -a resnet18 --resume resnet18.69978.pth --gpu 2 --backend snpe -e \
#     --train_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC \
#     --val_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC

# # FP32 val
# python main.py -a resnet18 -e --pretrained --gpu 0 -j 16 --not-quant

# # Calibration val
# python main.py -a resnet18 -e --pretrained --gpu 0 -j 16 --backend snpe

# python main_record.py -a resnet18 --epochs 1 --lr 1e-4 -b 128 --pretrained --gpu 1 --backend tensorrt -j 16 \
#     --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#     --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/


# python main_record.py -a resnet50 --epochs 100 --lr 1e-4 -b 128 --pretrained --backend snpe -j 32 -p 200 \
#     --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#     --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/


# python main_record.py -a resnet50  --pretrained --gpu 1 -j 16 -e -b 64 --not-quant \
#     --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#     --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/


# python main_record.py -a resnet50 --resume resnet50/model_best.pth.tar --deploy --backend snpe --gpu 1 \
#     --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#     --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/

python main_record.py -a mobilenet_v2 --epochs 10 --lr 1e-5 --optim adam --wd 0 -b 128 --pretrained --backend snpe -j 16 -p 500

# python main_record.py -a mobilenet_v2 --epochs 1 --lr 1e-5 --optim adam --wd 0 -b 64 --pretrained --backend snpe -j 16 \
#         --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#         --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/

# # eval qat model
# python main_record.py -a mobilenet_v2 -b 100 \
#         --resume ./mobilenet_v2/model_best.pth.tar \
#         --backend snpe \
#         -j 16 \
#         --gpu 1 \
#         -e

# # export onnx
# python main_record.py -a mobilenet_v2  --resume ./mobilenet_v2/model_best.pth.tar --deploy --backend snpe --gpu 1

# # eval float model
# python main_record.py -a mobilenet_v2 -b 256 --pretrained -j 8 \
#         --not-quant \
#         --gpu 1 \
#         -e \
#         --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#         --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/ \