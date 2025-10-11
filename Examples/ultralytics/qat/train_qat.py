from ultralytics import YOLO
from ultralytics.nn.modules.block import Chunk, Cat
from ultralytics.nn.modules.head import Detect, DetectPost
import torch
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.convert_deploy import convert_deploy
from mqbench.utils.state import enable_calibration, enable_quantization
# from preprocess import CalibDataset
from torch.utils.data import DataLoader


model = YOLO('yolov8n.yaml')
weights = torch.load("yolov8n_v2.pth")
model.load_state_dict(weights)
model.model.model_clear.train()

inp = torch.rand(1, 3, 640, 640).cuda()


extra_qconfig_dict = {
    'w_observer': 'MinMaxObserver',
    'a_observer': 'EMAMinMaxObserver',
    'w_fakequantize': 'FixedFakeQuantize',
    'a_fakequantize': 'FixedFakeQuantize',
}

leaf_module = (Chunk, Cat, DetectPost)

prepare_custom_config_dict = {
    'extra_qconfig_dict': extra_qconfig_dict,
    'leaf_module': leaf_module
}
model_tmp = prepare_by_platform(model.model.model_clear, BackendType.Tensorrt, prepare_custom_config_dict)

# print(model_tmp)


model.model.model_clear = model_tmp

model.model.model_clear.cuda()


enable_calibration(model.model)
# model.model(inp)

info = {
    "input_width": 640,
    "input_height": 640
}


exit(0)
dataset = CalibDataset("/mnt/share/yongyang/projects/mqb/L6/datasets/coco2017_yolo_labels/coco/images/train2017", 512, info)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for idx, data in enumerate(dataloader):
    print(f"calib : {idx}")
    model.model.model_clear(data.cuda())

enable_quantization(model.model)

# model.val(data='coco.yaml', cfg='custom.yaml')

# exit()

model.model.model_clear.train()
model.train(data='coco.yaml', cfg='custom.yaml')
# model.model.model_clear(inp)

# model.model.model_clear.train()

model.model.model_clear.eval()

model.model.model_clear(inp)

for n, m in model.model.model_clear.named_modules():
    if isinstance(m, DetectPost):
        m.export = True

convert_deploy(model.model.model_clear, BackendType.Tensorrt, {'x': [1, 3, 640, 640]}, model_name='yolo_trt_fixed_15')
