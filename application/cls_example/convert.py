import torch
import torchvision.models as models
import argparse
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization
def load_model(model_name, model_path, num_classes=1000):
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    extra_qconfig_dict = {
            'w_observer': 'MSEObserver',
            'a_observer': 'MSEObserver',
            'w_fakequantize': 'LearnableFakeQuantize',
            'a_fakequantize': 'LearnableFakeQuantize',
        }
    prepare_custom_config_dict = {'extra_qconfig_dict': extra_qconfig_dict}
    model = prepare_by_platform(model, BackendType.Tensorrt, True, prepare_custom_config_dict, None, True)
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') 
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def convert_to_onnx(model, input_size, onnx_path):
    dummy_input = torch.randn(1, 3, input_size, input_size)
    torch.onnx.export(model, dummy_input, onnx_path, opset_version=11, input_names=['input'], output_names=['output'])
    print(f"Model has been converted to ONNX and saved at {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model-name', type=str, default='mobilenet_v2', help='Name of the model (e.g., mobilenet_v2, resnet18)')
    parser.add_argument('--model-path', type=str, default='/home/ynz/new/MQBench/application/cls_example/checkpoints/checkpoint_5.pth', help='Path to the PyTorch model file')
    parser.add_argument('--onnx-path', type=str, default='mobilenet_v2', help='Path to save the ONNX model')
    parser.add_argument('--num-classes', type=int, default=200, help='Number of classes for the model')
    parser.add_argument('--input-size', type=int, default=224, help='Input size for the model')
    args = parser.parse_args()

    model = load_model(args.model_name, args.model_path, args.num_classes)
    convert_deploy(model, BackendType.Tensorrt, {'x': [1, 3, 224, 224]}, model_name='args.onnx_path')

if __name__ == "__main__":
    main()
