import os
import cv2
import numpy as np
import torch
import tfrecord
from tfrecord.tools import create_index
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

from tqdm import tqdm
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms


def cv2_loader(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class MyImageFolder(datasets.ImageFolder):
    
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


def main(train_dir, root):
    train_dataset = MyImageFolder(train_dir, loader=cv2_loader)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    
    total = 100
    index = 0
    file_num = 1
    batch_size = len(train_loader) // total
    print("Every file has {}".format(batch_size))
    record_file_prefix = root + '/train_{}.record'
    writer = tfrecord.TFRecordWriter(record_file_prefix.format(file_num))

    for img, lable in train_loader:
        h, w = img.shape[1:3]
        lable = int(lable[0])
        print(file_num, lable)
        img = img.squeeze(0).numpy().tobytes()
        writer.write({
            'image': (img, 'byte'),
            'lable': (lable, 'int'),
            'height': (int(h), 'int'),
            'width': (int(w), 'int')
        })
        index += 1
        if index == batch_size:
            writer.close()
            file_num += 1
            writer = tfrecord.TFRecordWriter(record_file_prefix.format(file_num))
            index = 0
    
    writer.close()

def create_record_index(root):
    tfrecord_paths = os.listdir(root)
    paths = [os.path.join(root, f) for f in tfrecord_paths if '.record' in f]

    for p in paths:
        index_file = p.replace('record', 'index')
        if os.path.isfile(index_file):
            continue
        create_index(p, index_file)
        print("Create index file {}".format(index_file))

def create_s_index(src, dst):
    create_index(src, dst)

if __name__ == '__main__':
    # train_dir = '/D2/olddata/DataCenter1/AlgData/Common/ImageNet/ILSVRC2012/Data/CLS-LOC/train'
    # root = '/D2/wzou/BenchmarkData/dataset/TFRecords/ImageNet/ILSVRC2012/train'
    # # main(train_dir, root)
    # create_record_index(root)
    import sys
    create_s_index(sys.argv[1], sys.argv[2])