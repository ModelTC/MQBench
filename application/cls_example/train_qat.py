import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization, disable_all
import argparse

BackendMap = {'tensorrt': BackendType.Tensorrt,
            'nnie': BackendType.NNIE,
            'ppl': BackendType.PPLW8A16,
            'snpe': BackendType.SNPE,
            'vitis': BackendType.Vitis,
            'tengine_u8': BackendType.Tengine_u8}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12354'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def build_model(model_name='mobilenet_v2', num_classes=200, model_path=None, device='cuda'):
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        input_size = 224
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        input_size = 224
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    device = torch.device(device)
    model = model.to(device)

    if model_path and os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False) 

    return model, input_size

def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, pred1 = outputs.topk(1, 1, True, True)
            _, pred5 = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            correct1 += pred1.eq(labels.view(-1, 1).expand_as(pred1)).sum().item()
            correct5 += pred5.eq(labels.view(-1, 1).expand_as(pred5)).sum().item()

    print(f"Loss: {val_loss/len(val_loader):.4f}, Acc1: {100.*correct1/total:.2f}%, Acc5: {100.*correct5/total:.2f}%")

def train(rank, world_size, model, input_size, args):
    setup(rank, world_size)
    extra_qconfig_dict = {
            'w_observer': 'MSEObserver',
            'a_observer': 'MSEObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
    prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
    model = prepare_by_platform(model, BackendMap[args.backend], prepare_custom_config_dict)

    if args.gpus:
        torch.cuda.set_device(args.gpus[rank])
        device = torch.device(f'cuda:{args.gpus[rank]}')
    else:
        device = torch.device(f'cuda:{rank}')

    transform_train = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'train'), transform=transform_train)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)

    val_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'val'), transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    cali_batch_size = 100
    cali_batch = 10
    cali_dataset = torch.utils.data.Subset(train_dataset, indices=torch.arange(cali_batch_size * cali_batch))
    cali_sampler = DistributedSampler(cali_dataset, num_replicas=world_size, rank=rank)
    cali_loader = DataLoader(cali_dataset, batch_size=cali_batch_size, sampler=cali_sampler, num_workers=4)

    model = model.to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss().to(rank)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=f'cuda:{rank}')
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    model.eval()
    enable_calibration(model)
    with torch.no_grad():
        for inputs, _ in tqdm(cali_loader, desc="Calibration"):
            inputs = inputs.to(device)
            model(inputs)
    enable_quantization(model)

    if rank == 0:
        evaluate(model, val_loader, device, criterion)

    for epoch in range(args.epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=rank != 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()

        if rank == 0:
            print(f"Train Epoch {epoch+1}, Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        if rank == 0:
            evaluate(model, val_loader, device, criterion)

        if rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_{epoch+1}.pth'))

    cleanup()


# 主函数
def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training with QAT')
    parser.add_argument('--data', metavar='DIR', required=True, help='path to dataset')
    parser.add_argument('--epochs', default=5, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--num_classes', default=200, type=int, help='number of classes')
    parser.add_argument('--gpus', default=[0, 1, 2, 3], type=int, nargs='+', help='gpus for distributed training')
    parser.add_argument('--model_name', default='mobilenet_v2', type=str, help='name of the model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--backend', default='tensorrt', type=str, help='backend type for QAT')
    parser.add_argument('--load-path', default='', type=str, metavar='PATH', help='path to latest checkpoint')

    args = parser.parse_args()
    world_size = len(args.gpus)  
    model_name = args.model_name  
    model, input_size = build_model(model_name, args.num_classes, args.load_path)
    model.share_memory()  
    torch.multiprocessing.spawn(train, args=(world_size, model, input_size, args), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()