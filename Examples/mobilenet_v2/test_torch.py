import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
from mbv2 import mobilenet_v2
import torch.utils.data as data
from torch.autograd import Variable
import time, os, copy, numpy as np
from dataset import get_dataset


def calc_acc(model):
    # test on val dataset 200 labels
    _, val_dataset, _ = get_dataset()
    dataloaders = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=16)

    running_corrects = 0.0
    for inputs, labels in tqdm(dataloaders):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    print(f"Accuracy : {running_corrects / len(val_dataset) * 100}%")



def export_onnx(model, onnx_path):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        
    x = torch.randn(1, 3, 224, 224).cuda()

    torch.onnx.export(
        model, 
        x, 
        onnx_path, 
        export_params=True, 
        opset_version=11
    )
    
    
if __name__ == "__main__":
    # model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobile_v2_best_model_200_labels.pth"
    # model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mbv2_fp16.pth"
    # model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mobilenet_v2-b0353104.pth"
    # model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mobilenet_v2.pth"
    # model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobile_v2_best_model_basic_tiny.pth"
    # model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/MQBench/Examples/models/mobilenet_v2.pth"

    model_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobile_v2_best_model_basic_tiny.pth"
    onnx_path = "/mnt/share_disk/bruce_trie/workspace/Quantizer-Tools/_outputs/models/mobile_v2_best_model_basic_tiny.onnx"

    model = torch.load(model_path)
    calc_acc(model)
    
    export_onnx(model, model_path)