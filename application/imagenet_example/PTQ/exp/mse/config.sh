arch=('r18' 'r50' 'mbv2' 'reg600' 'reg800')
name=('resnet18' 'resnet50' 'mobilenet_v2' 'regnetx_600m' 'regnetx_800m')
path=(
    'resnet18_imagenet.pth.tar'
    'resnet50_imagenet.pth.tar'
    'mobilenetv2.pth.tar'
    'spring_regnetx_600m.pth'
    'spring_regnetx_800m.pth'
)
for((i=0;i<5;i++))
do 
    quant_path=${arch[i]}_4_8
    mkdir -p $quant_path
    cp config.yaml $quant_path
    cp run.sh $quant_path
    cd $quant_path
    sed -re "s/type:([[:space:]]+)resnet18/type: ${name[i]}/" -i config.yaml
    sed -re "s/resnet18_imagenet.pth.tar/${path[i]}/" -i config.yaml
    cd ..
done
