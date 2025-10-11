from ultralytics import YOLO
import torch


model = YOLO('yolov8n.pt')
torch.save(model.state_dict(), "yolov8n.pth")
