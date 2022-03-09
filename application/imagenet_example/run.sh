# python main.py -a resnet18 --epochs 1 --lr 1e-4 -b 128 --pretrained --gpu 2 --backend snpe -j 16 

# python main.py -a mobilenet_v2 --epochs 1 --lr 1e-5 --wd 0 --optim adam  -b 64 --pretrained --gpu 0

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
#     --train_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC \
#     --val_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC


# python main.py -a mobilenet_v2 --epochs 1 --lr 1e-5 --wd 0 --optim adam  -b 64 --pretrained --gpu 2 --backend snpe -j 16 \
#     --train_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC \
#     --val_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC

# eval qat model
python main.py -a mobilenet_v2 \
    --resume ./mobilenet_v2/mobilenet_v2_acc_70.87.pth.tar \
    --gpu 2 \
    --backend snpe \
    -e \
    --train_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC \
    --val_data /D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC