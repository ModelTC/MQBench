import onnxruntime as ort
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm

def evaluate_onnx(model_path, input_size, batch_size, data_dir, device):
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 加载验证数据集
    val_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 检查可用的执行提供程序
    available_providers = ort.get_available_providers()
    if device == 'cuda' and 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # 创建 ONNX 运行时会话
    ort_session = ort.InferenceSession(model_path, providers=providers)

    # 验证
    val_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # ONNX 推理
            ort_inputs = {ort_session.get_inputs()[0].name: inputs.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            outputs = torch.tensor(ort_outs[0]).to(device)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, pred1 = outputs.topk(1, 1, True, True)
            _, pred5 = outputs.topk(5, 1, True, True)
            total += labels.size(0)
            correct1 += pred1.eq(labels.view(-1, 1).expand_as(pred1)).sum().item()
            correct5 += pred5.eq(labels.view(-1, 1).expand_as(pred5)).sum().item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Acc1: {100.*correct1/total:.2f}%, Acc5: {100.*correct5/total:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ONNX Image Classification Model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the ONNX model file')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the validation dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--input-size', type=int, default=224, help='Input size for the model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')

    args = parser.parse_args()

    evaluate_onnx(args.model_path, args.input_size, args.batch_size, args.data_dir, args.device)

if __name__ == "__main__":
    main()
