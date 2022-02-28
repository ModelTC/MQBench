
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

# python main.py -a resnet18 --epochs 1 --lr 1e-4 -b 128 --pretrained --gpu 2 --backend snpe -j 16 \
#     --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
#     --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/


python main_record.py -a mobilenet_v2 --epochs 1 --lr 1e-6 --optim adam  -b 64 --pretrained --gpu 2 --backend tensorrt -j 16 \
        --train_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train/ \
        --val_data /D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val/ \
        -p 100

# # eval qat model
# python main.py -a mobilenet_v2 --resume mobilenet_v2_acc_70.98_fixfake.pth.tar --gpu 2 --backend snpe -e \
#     --train_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC \
#     --val_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC