import os
import cv2
import numpy as np
import torch
import tfrecord
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset


from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms



# tfrecord_path = '/D2/olddata/DataCenter1/PrivateData/TFRecord-ImageNet/train_no_resize/imagenet_train-00000-00000.record'
# desc = {
#     "name": "byte",
#     "height": "int",
#     "width": "int",
#     "image": "byte",
#     "label": "int",
#     "labeltext": "byte",
#   }

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
# t = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ])


# def parse_example(record):
#     h, w = record['height'][0], record['width'][0]
#     img = np.fromstring(record['image'], np.uint8)
#     img = img.reshape(h, w, -1)[..., ::-1]
#     label = record['label'][0]
#     img = t(Image.fromarray(img))
#     return img, label - 1



# dataset = TFRecordDataset(tfrecord_path, None, desc, transform=parse_example)


# loader = torch.utils.data.DataLoader(dataset, batch_size=1)

# for d, l in loader:
#     print(d.shape, l)

import os
from tfrecord.tools import create_index

root = '/D2/olddata/DataCenter1/PrivateData/TFRecord-ImageNet/train_no_resize/'
tfrecord_paths = os.listdir(root)
paths = [os.path.join(root, f) for f in tfrecord_paths if '.record' in f]

for p in paths:
    basenemt = os.path.basename(p)
    basenemt = basenemt.replace('record', 'index')
    index_file = os.path.join('/home/corerain/MQBench/application/imagenet_index', basenemt)
    if os.path.isfile(index_file):
        continue
    create_index(p, index_file)