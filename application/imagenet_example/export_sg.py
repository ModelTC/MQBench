from selectors import EpollSelector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.utils.data.distributed
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_quantization, enable_calibration

import os
import sys
import json
import torchvision
import rbcompiler.api_v2 as rb_api
from mqbench.convert_deploy import convert_merge_bn
from efficientnet import EfficientNet, Conv2dStaticSamePadding

from extend_conv import (ConvStaticSamePaddingBn2dFusion, 
                         ConvStaticSamePaddingBn2d,
                         QConvStaticSamePaddingConvBn2d,
                         QConvStaticSamePadding2d,
                         fuse_conv_static_same_padding_bn)


if __name__ == '__main__':
    # model_name = sys.argv[1]
    # ckpt = sys.argv[2]
    model_name = 'torch_efficientnet_b0'
    ckpt = ''

    namemap = {}

    if model_name == "torch_shufflenet":
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'torch_squeezenet1_0':
        model = torchvision.models.squeezenet1_0(pretrained=True)
    elif model_name == 'torch_densenet121':
        model = torchvision.models.densenet121(pretrained=True)
    elif model_name == 'torch_efficientnet_b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)

    else:
        raise NotImplementedError
    
    sg = rb_api.gen_sg_from_pytorch(model, torch.rand(1,3,224,224))
    rb_api.save_sg(sg, '/home/wzou/rbquanttools/QAT/efficientnet_b0/efficientnet_b0.sg')
    exit(0)

    # from torch.fx import symbolic_trace
    # model = symbolic_trace(model)
    # print(model)

    model = prepare_by_platform(
        model, BackendType.SNPE,
        prepare_custom_config_dict={
            'extra_qconfig_dict': {
                'w_fakequantize': 'FixedFakeQuantize',
                'a_fakequantize': 'FixedFakeQuantize',
                'w_qscheme': {'symmetry': True, 'per_channel': True, 'pot_scale': False, 'bit': 8},
                'a_qscheme': {'symmetry': False, 'per_channel': False, 'pot_scale': False, 'bit': 8}
            },
            'leaf_module': [Conv2dStaticSamePadding],
            'extra_fuse_dict': {
                'additional_fusion_pattern': {
                    (nn.BatchNorm2d, Conv2dStaticSamePadding): ConvStaticSamePaddingBn2dFusion
                },
                'additional_fuser_method_mapping': {
                    (Conv2dStaticSamePadding, torch.nn.BatchNorm2d): fuse_conv_static_same_padding_bn
                },
                'additional_qat_module_mapping': {
                    ConvStaticSamePaddingBn2d: QConvStaticSamePaddingConvBn2d,
                    Conv2dStaticSamePadding: QConvStaticSamePadding2d
                }
            },
            'extra_quantizer_dict': {
                'additional_module_type': (QConvStaticSamePadding2d, )
            }
        }
    )
    # print(model)
    # # print(model.code)
    # exit(0)

    # backbone = prepare_by_platform(model.backbone, BackendType.SNPE)
    # print(backbone.code)
    # exit(0)

    enable_calibration(model)
    model.cuda()

    # print("=> loading checkpoint '{}'".format(ckpt))
    # checkpoint = torch.load(ckpt, map_location='cuda:0')
    # state_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()

    # if 'module.' in list(state_dict.keys())[0] and 'module.' not in list(model_dict.keys())[0]:
    #     for k in list(state_dict.keys()):
    #         state_dict[k[7:]] = state_dict.pop(k)

    # model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        for i in range(2):
            x = torch.rand(1,3,224,224).cuda()
            model(x)

    enable_quantization(model)

    # convert_deploy(model.eval(), BackendType.SNPE, input_shape_dict={'data': [1, 3, 224, 224]}, output_path='./qat_sgs')
    # exit(0)

    convert_merge_bn(model.eval())
    # model.to(torch.device('cpu'))
    dumpy_input = torch.rand(1,3,224,224).cuda()
    output_dir = './qat_sgs'
    export_sg_file = os.path.join(output_dir, '{}.sg'.format(model_name))
    clip_range_file = os.path.join(output_dir, '{}_act_clip.json'.format(model_name))
    sg = rb_api.gen_sg_from_pytorch(model.eval(), dumpy_input, clip_range_file=clip_range_file)
    rb_api.save_sg(sg, export_sg_file)

    # export 8bit sg
    with open(clip_range_file, 'r') as f:
        clip_ranges = json.loads(f.read())

    # qconfig = {'quant_ops': {'DepthWiseConv2D': {'per_channel': False}}}
    qconfig = {'quant_ops': {'Conv2D': {'per_channel': True}}}
    qsg = rb_api.gen_quant_sg_from_clip_ranges(sg, clip_ranges, qconfig)
    rb_api.save_sg(qsg, export_sg_file.replace('.sg', '_8bit.sg'))
