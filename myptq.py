import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from mqbench.convert_deploy import convert_deploy
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.utils.state import enable_calibration, enable_quantization
from mqbench.advanced_ptq import ptq_reconstruction
from easydict import EasyDict
import os
import platform
import numpy as np
from collections import defaultdict

# ---------------------------
# Configuration
# ---------------------------
RANDOM_SEED = 42
SAMPLES_PER_CLASS = 1  # Adjust this to change calibration samples per class

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------
# GPU Verification
# ---------------------------
print("=" * 50)
print("GPU VERIFICATION")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN available: {torch.backends.cudnn.enabled}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current GPU: {torch.cuda.current_device()}")
else:
    print("CUDA NOT DETECTED")
print("=" * 50 + "\n")

# ---------------------------
# Device configuration
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ---------------------------
# Data preparation
# ---------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Detect dataset paths based on WSL or Windows
def get_dataset_paths():
    if 'microsoft' in platform.uname().release.lower() or 'WSL' in platform.uname().release:
        val_path = '/mnt/c/Users/Qi Yang/Desktop/FYP/ImageNet-ILSVRC2012/imagenet-val/'
        train_path = '/mnt/c/Users/Qi Yang/Desktop/FYP/ImageNet-ILSVRC2012/imagenet_subtrain/'
    else:
        val_path = 'C:/Users/Qi Yang/Desktop/FYP/ImageNet-ILSVRC2012/imagenet-val/'
        train_path = 'C:/Users/Qi Yang/Desktop/FYP/ImageNet-ILSVRC2012/imagenet_subtrain/'
    return val_path, train_path

val_dataset_path, train_dataset_path = get_dataset_paths()
print(f"Validation dataset path: {val_dataset_path}")
print(f"Training subset path: {train_dataset_path}")

if not os.path.exists(val_dataset_path):
    raise FileNotFoundError(f"Validation dataset path not found: {val_dataset_path}")
if not os.path.exists(train_dataset_path):
    raise FileNotFoundError(f"Training dataset path not found: {train_dataset_path}")

# Load validation dataset (for evaluation)
val_dataset = datasets.ImageFolder(
    root=val_dataset_path,
    transform=transform
)

# Load training subset dataset (for calibration)
train_dataset = datasets.ImageFolder(
    root=train_dataset_path,
    transform=transform
)

print(f"Validation dataset: {len(val_dataset)} images, {len(val_dataset.classes)} classes")
print(f"Training subset: {len(train_dataset)} images, {len(train_dataset.classes)} classes\n")

# ---------------------------
# Stratified Sampling for Calibration
# ---------------------------
def stratified_sampling(dataset, samples_per_class, seed=42):
    """
    Perform stratified sampling to select a fixed number of samples per class.
    
    Args:
        dataset: ImageFolder dataset
        samples_per_class: Number of samples to select from each class
        seed: Random seed for reproducibility
    
    Returns:
        List of selected indices
    """
    np.random.seed(seed)
    
    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)
    
    # Sample from each class
    selected_indices = []
    for class_label in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_label]
        if len(indices) < samples_per_class:
            print(f"WARNING: Class {class_label} has only {len(indices)} samples, "
                  f"but {samples_per_class} requested. Using all available samples.")
            selected_indices.extend(indices)
        else:
            # Randomly select samples_per_class samples from this class
            selected = np.random.choice(indices, samples_per_class, replace=False)
            selected_indices.extend(selected.tolist())
    
    # Shuffle the selected indices
    np.random.shuffle(selected_indices)
    
    return selected_indices

print("=" * 50)
print("STRATIFIED CALIBRATION SAMPLING")
print("=" * 50)
print(f"Samples per class: {SAMPLES_PER_CLASS}")
print(f"Random seed: {RANDOM_SEED}")
print("Performing stratified sampling...")

cali_indices = stratified_sampling(train_dataset, SAMPLES_PER_CLASS, seed=RANDOM_SEED)
cali_dataset = Subset(train_dataset, cali_indices)

print(f"Selected {len(cali_indices)} calibration samples")
print(f"Expected: {len(train_dataset.classes)} classes * {SAMPLES_PER_CLASS} samples = "
      f"{len(train_dataset.classes) * SAMPLES_PER_CLASS} samples\n")

# Validation dataset will be used entirely for evaluation
eval_dataset = val_dataset

# ---------------------------
# Data loaders
# ---------------------------
batch_size = 32
cali_loader = DataLoader(
    cali_dataset,
    batch_size=batch_size,
    shuffle=False,  # Already shuffled in stratified sampling
    num_workers=4,
    pin_memory=True,
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

print(f"Calibration dataset: {len(cali_dataset)} images")
print(f"Evaluation dataset: {len(eval_dataset)} images\n")

# ---------------------------
# Quantization Configuration
# ---------------------------
extra_qconfig_dict = {
    'w_observer': 'EMAMSEObserver',
    'a_observer': 'EMAMSEObserver',
    'w_fakequantize': 'AdaRoundFakeQuantize',
    'a_fakequantize': 'LearnableFakeQuantize',
    'w_qscheme': {
        'bit': 4,
        'symmetry': True,
        'per_channel': True,
        'pot_scale': False # pot_scale is not a usual setting, which turns on non-uniform quantization
    },
    'a_qscheme': {
        'bit': 4,
        'symmetry': True,
        'per_channel': False, # we can not use per-channel quantization for activation which prohibits low-bit inference acceleration
        'pot_scale': False
    }
}

# ---------------------------
# PTQ Reconstruction Config
# ---------------------------
ptq_reconstruction_config = EasyDict({
    'pattern': 'layer',
    'scale_lr': 4.0e-5,
    'warm_up': 0.2,
    'weight': 0.01,
    'max_count': 10000,
    'b_range': [20, 2],
    'keep_gpu': True,
    'round_mode': 'learned_hard_sigmoid',
    'prob': 1.0,
})

# ---------------------------
# Backend
# ---------------------------
backend = BackendType.Academic

# ---------------------------
# Helper function for evaluation
# ---------------------------
def evaluate_model(model, loader, device, name="Model"):
    """Evaluate model accuracy on given data loader."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 50 == 0 or (i + 1) == len(loader):
                current_acc = 100 * correct / total
                print(f"  {name}: Batch {i + 1}/{len(loader)}, "
                      f"Current Accuracy: {current_acc:.2f}%")
    
    accuracy = 100 * correct / total
    return accuracy

# ---------------------------
# Evaluate FP32 Baseline
# ---------------------------
print("=" * 50)
print("FP32 BASELINE EVALUATION")
print("=" * 50)
fp32_model = models.resnet18(pretrained=True)
fp32_model.to(device)
fp32_model.eval()
fp32_accuracy = evaluate_model(fp32_model, eval_loader, device, "FP32 Model")
print(f"\nFP32 Baseline Accuracy: {fp32_accuracy:.2f}%\n")

# Clean up FP32 model to save memory
del fp32_model
torch.cuda.empty_cache()

# ---------------------------
# Prepare Model for Quantization
# ---------------------------
print("=" * 50)
print("QUANTIZATION PIPELINE")
print("=" * 50)
print("Initializing model...")
model = models.resnet18(pretrained=True)
model.to(device)
model.eval()

print("Preparing model for quantization...")
prepare_custom_config = {
    'extra_qconfig_dict': extra_qconfig_dict
}

model = prepare_by_platform(
    model, 
    backend, 
    prepare_custom_config_dict=prepare_custom_config
)

model.to(device)
print("Quantization preparation completed!\n")

# ---------------------------
# Calibration & Data Collection
# ---------------------------
print("=" * 50)
print("CALIBRATION & DATA COLLECTION")
print("=" * 50)
print("Starting calibration process...")
enable_calibration(model)
stacked_tensor = []

with torch.no_grad():
    for i, (images, _) in enumerate(cali_loader):
        images = images.to(device)
        model(images)
        stacked_tensor.append(images)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(cali_loader):
            print(f"  Processed: {i + 1}/{len(cali_loader)} batches")

enable_quantization(model)
model.eval()
print(f"Calibration completed! Collected {len(stacked_tensor)} batches for PTQ reconstruction\n")

# ---------------------------
# PTQ Reconstruction
# ---------------------------
print("=" * 50)
print("PTQ RECONSTRUCTION (AdaRound)")
print("=" * 50)
print(f"Configuration:")
print(f"  Pattern: {ptq_reconstruction_config.pattern}")
print(f"  Max iterations: {ptq_reconstruction_config.max_count}")
print(f"  Weight regularization: {ptq_reconstruction_config.weight}")
print(f"  Warm-up ratio: {ptq_reconstruction_config.warm_up}")
print(f"  Temperature range: {ptq_reconstruction_config.b_range}")
print(f"  QDrop probability: {ptq_reconstruction_config.prob}")
print("\nStarting reconstruction...")

try:
    model = ptq_reconstruction(model, stacked_tensor, ptq_reconstruction_config)
    print("\nPTQ reconstruction completed successfully!\n")
except Exception as e:
    print(f"\nWARNING: PTQ reconstruction failed with error: {e}")
    print("Continuing with post-calibration quantized model...\n")

# Clean up calibration data
del stacked_tensor
torch.cuda.empty_cache()

# ---------------------------
# Evaluate Quantized Model
# ---------------------------
print("=" * 50)
print("QUANTIZED MODEL EVALUATION")
print("=" * 50)
enable_quantization(model)
model.eval()
quantized_accuracy = evaluate_model(model, eval_loader, device, "Quantized Model")
print(f"\nQuantized Model Accuracy: {quantized_accuracy:.2f}%\n")

# ---------------------------
# Results Summary
# ---------------------------
print("=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Dataset Configuration:")
print(f"  Calibration source: Training subset (stratified)")
print(f"  Samples per class: {SAMPLES_PER_CLASS}")
print(f"  Total calibration samples: {len(cali_dataset)}")
print(f"  Evaluation source: Validation set")
print(f"  Total evaluation samples: {len(eval_dataset)}")
print(f"  Random seed: {RANDOM_SEED}")
print()
print(f"Quantization Configuration:")
print(f"  Weight bits: {extra_qconfig_dict['w_qscheme']['bit']}")
print(f"  Activation bits: {extra_qconfig_dict['a_qscheme']['bit']}")
print(f"  Weight observer: {extra_qconfig_dict['w_observer']}")
print(f"  Activation observer: {extra_qconfig_dict['a_observer']}")
print(f"  Weight quantizer: {extra_qconfig_dict['w_fakequantize']}")
print(f"  Activation quantizer: {extra_qconfig_dict['a_fakequantize']}")
print()
print(f"Accuracy Results:")
print(f"  FP32 Baseline:      {fp32_accuracy:.2f}%")
print(f"  Quantized Model:    {quantized_accuracy:.2f}%")
print(f"  Accuracy Drop:      {fp32_accuracy - quantized_accuracy:.2f}%")
print(f"  Relative Drop:      {((fp32_accuracy - quantized_accuracy) / fp32_accuracy * 100):.2f}%")
print("=" * 50 + "\n")

# ---------------------------
# Model Deployment Conversion
# ---------------------------
print("=" * 50)
print("DEPLOYMENT CONVERSION")
print("=" * 50)
print("Converting quantized model for deployment...")
model.cpu()
input_shape = {'data': [10, 3, 224, 224]}

try:
    convert_deploy(model, backend, input_shape)
    print("Deployment conversion completed successfully!")
except Exception as e:
    print(f"WARNING: Deployment conversion failed: {e}")
    print("Model is still usable for inference in current framework.")

print("\n" + "=" * 50)
print("QUANTIZATION PIPELINE COMPLETED")
print("=" * 50)