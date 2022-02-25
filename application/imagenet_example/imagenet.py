import enum
import os
import time
import functools
import numpy as np
from PIL import Image

import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset



class MySubset(torch.utils.data.IterableDataset):

    def __init__(self, dataset, end):
        super(MySubset).__init__()
        self.dataset = dataset
        self.end = end

    def __iter__(self):
        def iters():
            index = 0
            for item in self.dataset:
                yield item
                index += 1
                if index == self.end:
                    break
        return iters()


class MyImageFolder(ImageFolder):
    
    @staticmethod
    def make_dataset(
        directory,
        class_to_idx,
        extensions = None,
        is_valid_file = None,
    ):
        instances = []
        directory = os.path.expanduser(directory)
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            #if not os.path.isdir(target_dir):
            #    continue
            fnames = [f for f in os.listdir(target_dir) if '.JPEG' in f]
            for fname in fnames:
                instances.append((os.path.join(target_dir, fname), class_index))
            # for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            #     for fname in sorted(fnames):
            #         path = os.path.join(root, fname)
            #         if is_valid_file(path):
            #             item = path, class_index
            #             instances.append(item)
        return instances

# train_dataset = MyImageFolder('/D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC/train', transform=t)

def parse_example(record, transform=None, val=False):
    h, w = record['height'][0], record['width'][0]
    img = np.frombuffer(record['image'], np.uint8)
    img = img.reshape(h, w, -1)
    if val:
        img = np.ascontiguousarray(img[..., ::-1])
    label = record.get('label', 'lablel')[0]
    if transform is not None:
        img = transform(Image.fromarray(img))
    return img, label


def get_train_dataset(root):
    desc = {
        "image": "byte",
        "height": "int",
        "width": "int",
        "lable": "int"
    }
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_transform = functools.partial(parse_example, transform=transform)
    tfrecord_paths = os.listdir(root)
    tfrecord_paths = [f for f in tfrecord_paths if '.record' in f]
    splits = {f.split('.')[0]: 0.5 for f in tfrecord_paths}

    tfrecord_pattern = root + '/{}.record'
    index_pattern = root + '/{}.index'
    train_dataset = MultiTFRecordDataset(
        tfrecord_pattern, index_pattern, splits, desc, transform=dataset_transform, infinite=False)
    return train_dataset


def get_val_dataset(root):
    desc = {
        "image": "byte",
        "height": "int",
        "width": "int",
        "label": "int"
    }
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_transform = functools.partial(parse_example, transform=transform, val=True)
    val_path = os.path.join(root, 'imagenet_val.tfrecord')
    val_index = os.path.join(root, 'imagenet_val.index')
    val_dataset = TFRecordDataset(val_path, val_index, desc, transform=dataset_transform)
    return val_dataset


def get_calib_loader(root):
    desc = {
        "image": "byte",
        "height": "int",
        "width": "int",
        "lable": "int"
    }
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset_transform = functools.partial(parse_example, transform=transform)
    tfrecord_path = os.path.join(root, 'train_2.record')
    index_path = os.path.join(root, 'train_2.index')
    dataset = TFRecordDataset(tfrecord_path, index_path, desc, transform=dataset_transform)
    cali_dataset = MySubset(dataset, 100)
    cali_loader = torch.utils.data.DataLoader(cali_dataset, batch_size=10, shuffle=False,
                                              num_workers=0, pin_memory=False)
    return cali_loader



if __name__ == '__main__':
    # loader = get_calib_loader('/D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train')
    # dataset = get_val_dataset('/D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/val')
    # loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False,
    #                                      num_workers=4, pin_memory=False)
    train_dataset = get_train_dataset('/D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train')
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False,
                                         num_workers=8, pin_memory=False)
    print(len(loader))
    end = time.time()
    for d, l in loader:
        print(d.shape, time.time() - end)
        end = time.time()