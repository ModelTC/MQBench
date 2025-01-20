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
import argparse

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def build_model(model_name='mobilenet_v2', num_classes=200):
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
    return model, input_size

def train(rank, world_size, model, input_size, args):
    setup(rank, world_size)

    if args.gpus:
        torch.cuda.set_device(args.gpus[rank])
        device = torch.device(f'cuda:{args.gpus[rank]}')
    else:
        device = torch.device(f'cuda:{rank}')

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
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

    model = model.to(device)
    model = DDP(model, device_ids=[device])
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

            print(f"Validation Epoch {epoch+1}, Loss: {val_loss/len(val_loader):.4f}, Acc1: {100.*correct1/total:.2f}%, Acc5: {100.*correct5/total:.2f}%")

        if rank == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpoint_{epoch+1}.pth'))


    cleanup()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Image Classification Training')
    parser.add_argument('--data', metavar='DIR', required=True, help='path to dataset')
    parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--num_classes', default=200, type=int, help='number of classes')
    parser.add_argument('--gpus', default=[0, 1, 2, 3], type=int, nargs='+', help='gpus for distributed training')
    parser.add_argument('--model_name', default='mobilenet_v2', type=str, help='name of the model')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--checkpoint_dir', default='./', type=str, help='directory to save checkpoints')

    args = parser.parse_args()
    world_size = len(args.gpus) 
    model_name = args.model_name 
    model, input_size = build_model(model_name, args.num_classes)
    torch.multiprocessing.spawn(train, args=(world_size, model, input_size, args), nprocs=world_size, join=True)
if __name__ == "__main__":
    main()