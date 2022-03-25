
# # FP32 val
# python main_record.py -a resnet18 -e --pretrained --gpu 0 -j 16 --not-quant 
# python main_record.py -a squeezenet1_0 -e --pretrained --gpu 0 -j 16 --not-quant -b 128

# export ptq sg
# python main_record.py -a efficientnet_b0 --pretrained --gpu 2 -j 16 -b 128 --deploy --backend snpe
# python main_record.py -a squeezenet1_0 --pretrained --gpu 0 -j 16 -b 128 --backend snpe --deploy

# eval ptq
# python main_record.py -a squeezenet1_0 --pretrained --gpu 0 -j 16 -b 128 -e --backend snpe

# train
# python main_record.py -a efficientnet_b0 --pretrained -j 16 -b 64 --backend snpe --lr 1e-4 --epochs 10

# python main_record.py -a squeezenet1_0 --pretrained -j 16 -b 64 --backend snpe --lr 1e-5 --epochs 10 --gpu 4 --weight-decay 0.0
python main_record.py -a squeezenet1_0 \
    --resume ./squeezenet1_0/squeezenet1_0_acc_57.76_epoch_6.pth.tar \
    -j 16 -b 64 --backend snpe --lr 1e-5 --epochs 10 --gpu 4 --weight-decay 0.0

# python main_record.py -a squeezenet1_0 \
#     --resume ./squeezenet1_0/squeezenet1_0_acc_56.30_epoch_0.pth.tar \
#     -j 16 -b 128 --backend snpe --lr 1e-4 --epochs 10

# export qat sg
# python main_record.py -a efficientnet_b0 \
#     --resume efficientnet_b0/model_best.pth.tar \
#     --gpu 3 \
#     -j 16 \
#     -b 128 \
#     --deploy \
#     --backend snpe