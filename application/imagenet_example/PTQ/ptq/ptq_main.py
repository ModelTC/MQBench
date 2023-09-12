import torchvision.models as models
import numpy as np
import time
import argparse
from data.imagenet import load_data
from models import load_model
from utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', metavar='DIR',
                    help='path to dataset', required=True)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--backend', type=str, choices=['academic', 'tengine_u8', 'tensorrt', 'nnie', 'ppl', 'snpe', 'sophgo_tpu', 'openvino', 'tensorrt_nlp'], default='sophgo_tpu')
parser.add_argument('--cali-batch-num', default=16, type=int,
                    metavar='N', help='set calibration batch num (default: 16)')
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--quantize_type', metavar='DIR',
                    help='set quantize_type', type=str, default='naive_ptq')
parser.add_argument('--deploy', action='store_true')

BackendMap = {
    'academic': BackendType.Academic,
    'sophgo_tpu': BackendType.Sophgo_TPU,
    'nnie': BackendType.NNIE,
    'tensorrt_nlp': BackendType.Tensorrt_NLP,
    'ppl': BackendType.PPLW8A16,
    'openvino': BackendType.OPENVINO,
    'snpe': BackendType.SNPE,
    'vitis': BackendType.Vitis,
    'tengine_u8': BackendType.Tengine_u8,
    'tensorrt': BackendType.Tensorrt
}


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data

def get_quantize_model(model, args):
    backend_type = BackendType.Academic if not hasattr(
        args, 'backend') else BackendMap[args.backend]

    if backend_type == BackendType.Academic:
        extra_prepare_dict = {
            "extra_qconfig_dict": {
                                    'w_observer': 'MinMaxObserver',
                                    'a_observer': 'EMAMinMaxObserver',
                                    "w_fakequantize": "FixedFakeQuantize",
                                    "a_fakequantize": "FixedFakeQuantize",
                                    'w_qscheme': {  'bit': 8,
                                                    'symmetry': False,
                                                    'per_channel': True,
                                                    'pot_scale': False },
                                    'a_qscheme': {  'bit': 8,
                                                    'symmetry': False,
                                                    'per_channel': False,
                                                    'pot_scale': False }                  
                                }
        }
    elif backend_type == BackendType.Sophgo_TPU:
        extra_prepare_dict = {
            "extra_qconfig_dict": { 
                                    'w_observer': 'MinMaxObserver',
                                    'a_observer': 'KLDObserver',}}
    else:
        extra_prepare_dict = {}
    return prepare_by_platform(
        model, backend_type, prepare_custom_config_dict=extra_prepare_dict)


def deploy(model, args):
    backend_type = BackendType.Academic if not hasattr(
        args, 'backend') else BackendMap[args.backend]
    output_path = './' if not hasattr(
        args, 'output_path') else args.output_path
    model_name = args.arch
    deploy_to_qlinear = False if not hasattr(
        args, 'deploy_to_qlinear') else args.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)

def main():
    time_start = time.time()
    args = parser.parse_args()

    # set output path
    if args.output_path is None:
        args.output_path = './'
    args.output_path=os.path.join(args.output_path, args.arch) 
    os.system('mkdir -p {}'.format(args.output_path))

    # set seed first
    if args.seed is not None:
        seed_all(args.seed)

    # create_model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
    model.cuda()

    # load_data
    train_loader, val_loader = load_data(path=args.data_path , batch_size=args.batch_size)
    
    # evaluate
    evaluate(val_loader, model)

    # get quantize model
    model = get_quantize_model(model, args)
    model.cuda()
    print('>>>>>model after insert fake quantization node: ', model)

    # ptq
    if args.quantize_type == 'advanced_ptq':
        print('begin calibration now!')
        cali_data = load_calibrate_data(train_loader, cali_batchsize=args.cali_batch_num)
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        import torch
        with torch.no_grad():
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                model(batch.cuda())
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[0].cuda())
        print('begin advanced PTQ now!')
        if hasattr(args, 'reconstruction'):
            model = ptq_reconstruction(
                model, cali_data, args.reconstruction)
        enable_quantization(model)
        evaluate(val_loader, model)
        if args.deploy:
            deploy(model, args)
    elif args.quantize_type == 'naive_ptq':
        print('begin calibration now!')
        cali_data = load_calibrate_data(train_loader, cali_batchsize=args.cali_batch_num)
        from mqbench.utils.state import enable_quantization, enable_calibration
        model.eval() 
        enable_calibration(model)
        for batch in cali_data:
            model(batch.cuda())
        print('begin quantization now!')
        enable_quantization(model)
        evaluate(val_loader, model)
        if args.deploy:
            deploy(model, args)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")
    time_end = time.time()
    print('totally time is ', time_end-time_start)

if __name__ == '__main__':
    main()