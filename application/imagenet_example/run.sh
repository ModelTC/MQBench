python main_record.py -a efficientnet_b0 \
    --resume efficientnet_b0/model_best.pth.tar \
    --gpu 5 \
    -j 16 \
    -b 128 \
    --deploy \
    --backend snpe