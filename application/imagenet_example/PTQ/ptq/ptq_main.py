import sys
import os
import time
sys.path.append(os.path.abspath('.'))

import torch
import torchvision.models as models
import numpy as np
import time
import argparse
from data.imagenet import load_data
from models import load_model
from utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform
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
parser.add_argument('--chip', type=str, choices=['A2', 'BM1684X', 'SG2260', 'academic'], default='SG2260')
parser.add_argument('--quantmode', type=str, choices=['weight_activation', 'weight_only'], default='weight_activation')
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
parser.add_argument('--cpu', action='store_true',
                    help='use cpu to quant')
parser.add_argument('--fp8', action='store_true')

def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    return cali_data

def get_quantize_model(model, args):

    extra_prepare_dict = {
        'quant_dict': {
                        'chip': args.chip,
                        'quantmode': args.quantmode,
                        'strategy': 'CNN',
                       },
    }

    if args.fp8:
        extra_prepare_dict["extra_qconfig_dict"] = {
                                'w_observer': 'MinMaxObserver',
                                'a_observer': 'EMAMinMaxObserver',
                                "w_fakequantize": 'E5M2FakeQuantize',
                                "a_fakequantize": 'E5M2FakeQuantize',
                                # "a_fakequantize": 'LearnableFakeQuantize',
                                'w_qscheme': {  'bit': 8,
                                                'symmetry': True,
                                                'per_channel': False,
                                                'pot_scale': False },
                                'a_qscheme': {  'bit': 8,
                                                'symmetry': True,
                                                'per_channel': False,
                                                'pot_scale': False }
                            }

    # For mobilenet_v3_small, we set some fakequant node only observe
    if "mobilenet_v3" in args.arch:
        extra_prepare_dict["extra_quantizer_dict"] = {'module_only_enable_observer': [
                                                                'features.0.0.weight_fake_quant',
                                                                'features.1.block.0.0.weight_fake_quant',
                                                                'features.1.block.1.fc1.weight_fake_quant',
                                                                'features.1.block.1.fc2.weight_fake_quant',
                                                                'features.1.block.2.0.weight_fake_quant',
                                                                'features.2.block.0.0.weight_fake_quant',
                                                                'features.2.block.1.0.weight_fake_quant',
                                                                'features.2.block.2.0.weight_fake_quant',

                                                                'x_post_act_fake_quantizer',
                                                                'features_0_0_post_act_fake_quantizer',
                                                                'features_0_2_post_act_fake_quantizer',
                                                                'features_1_block_0_0_post_act_fake_quantizer',
                                                                'features_1_block_1_avgpool_post_act_fake_quantizer',
                                                                'features_1_block_1_fc1_post_act_fake_quantizer',
                                                                'features_1_block_1_fc2_post_act_fake_quantizer',
                                                                'features_1_block_1_scale_activation_post_act_fake_quantizer',
                                                                'mul_post_act_fake_quantizer',
                                                                'features_1_block_2_0_post_act_fake_quantizer',
                                                                'features_2_block_0_0_post_act_fake_quantizer',
                                                                'features_2_block_1_0_post_act_fake_quantizer',
                                                                ]
                                                            }

    return prepare_by_platform(model, prepare_custom_config_dict=extra_prepare_dict)

def deploy(model, args):
    net_type = 'CNN'
    output_path = './' if not hasattr(
        args, 'output_path') else args.output_path
    model_name = args.arch
    deploy_to_qlinear = False if not hasattr(
        args, 'deploy_to_qlinear') else args.deploy_to_qlinear

    convert_deploy(model, net_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)

def main():
    time_start = time.time()
    args = parser.parse_args()

    # set output path
    if args.output_path is None:
        args.output_path = './'
    if args.fp8:
        args.output_path=os.path.join(args.output_path, args.arch+'_fp8')
    else:
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
    if not args.cpu:
        model.cuda()

    # load_data
    train_loader, val_loader = load_data(path=args.data_path , batch_size=args.batch_size)
    
    # evaluate
    evaluate(val_loader, model)

    # get quantize model
    model = get_quantize_model(model, args)
    if not args.cpu:
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
                if not args.cpu:
                    model(batch.cuda())
                else:
                    model(batch)
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            if not args.cpu:
                model(cali_data[0].cuda())
            else:
                model(cali_data[0])
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
            if not args.cpu:
                model(batch.cuda())
            else:
                model(batch)
        print('begin quantization now!')
        enable_quantization(model)
        print('begin eval now!')
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