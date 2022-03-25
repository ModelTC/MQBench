

python main_a.py -a efficientnet_b0 \
    --pretrained \
    -j 16 \
    -b 64 \
    --backend snpe \
    --lr 0.00625 \
    --epochs 10 \
    --momentum 0.89 \
    --weight-decay 4.50e-05