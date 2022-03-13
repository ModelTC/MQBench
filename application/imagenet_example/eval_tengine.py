import argparse
import os
import time

import torch
from tengine import tg
import numpy as np
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from main import (
    accuracy,
    AverageMeter
)

IMG_SIZE = 224

parser = argparse.ArgumentParser(description='classification validation with pytengine')
parser.add_argument('--dataset', metavar='DIR',
                    help='path to dataset', required=True)
parser.add_argument('-m', '--model', required=True, type=str,
                        help='path to tengine model file.')
parser.add_argument('-b', '--batch-size', type=int, default=64)


def infer(engine, inputs: np.ndarray) -> np.ndarray:
    outputs = []
    input_tensor = engine.getInputTensor(0, 0)
    input_quant_param = input_tensor.getQuantParam(1)
    scale = input_quant_param[0][0]
    zp = input_quant_param[1][0]
    # compatible with quantized and float-point models
    if scale != 0:
        inputs = (inputs / scale + zp).round().clip(0, 255).astype(np.uint8)
    for img in inputs:
        input_tensor.buf = img
        engine.run(1) # 1 is blocking
        output_tensor = engine.getOutputTensor(0, 0)
        outputs.append(np.array(output_tensor.buf).reshape(1, -1))

    return np.concatenate(outputs)

def validation():
    args = parser.parse_args()

    # load model and prerun
    graph = tg.Graph(None, 'tengine', args.model)
    input_tensor = graph.getInputTensor(0, 0)
    dims = [1, 3, IMG_SIZE, IMG_SIZE]
    input_tensor.shape = dims
    graph.preRun()

    # prepare dataset
    valdir = os.path.join(args.dataset, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    # doing eval
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # evaluate
    for index, (images, target) in enumerate(val_loader):
        images = images.detach().numpy()
        output = infer(graph, images)
        output = torch.from_numpy(output)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.shape[0])
        top5.update(acc5[0], images.shape[0])

        if index % 100 == 0:
            print(f' {index} ==> * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print(f' Final ==> * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))


if __name__ == '__main__':
    validation()
