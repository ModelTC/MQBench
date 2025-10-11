import os
import random
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import json
from common.configs import get_cfg_defaults

cfg = get_cfg_defaults()

#200类，每类随机选5个
def get_calib_data_path():
    img_paths = []
    data_root = f"{cfg.SYSTEM.imagenet_200_dir}/val/"
    data_info = pd.read_table(data_root + "val_annotations.txt")
    grouped = data_info.groupby(data_info.columns[1])
    classes = list(grouped.groups.keys())
    for cls in classes:
        group_imgs = grouped.get_group(cls).iloc[:, 0].tolist()
        random.shuffle(group_imgs)
        img_paths += group_imgs[:5]

    return img_paths

# print(get_calib_data_path())

def Preprocess(img):
    transforms_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )
    img = transforms_val(img)
    return img

# For TRT
class CalibDataLoader:
    def __init__(self, batch_size, calib_count):
        self.data_root = f"{cfg.SYSTEM.imagenet_200_dir}/val/images/"

        self.index = 0
        self.batch_size = batch_size
        self.calib_count = calib_count
        self.image_list = get_calib_data_path()
        self.calibration_data = np.zeros(
            (self.batch_size, 3, 224, 224), dtype=np.float32
        )

    def reset(self):
        self.index = 0

    def next_batch(self):
        if self.index < self.calib_count:
            for i in range(self.batch_size):
                image_path = self.image_list[i + self.index * self.batch_size]
                image = Image.open(self.data_root + image_path).convert("RGB")
                image = Preprocess(image)
                self.calibration_data[i] = image
            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.calib_count


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file=""):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.data_loader = data_loader
        self.d_input = cuda.mem_alloc(self.data_loader.calibration_data.nbytes)
        self.cache_file = cache_file
        data_loader.reset()

    def get_batch_size(self):
        return self.data_loader.batch_size

    def get_batch(self, names):
        batch = self.data_loader.next_batch()
        if not batch.size:
            return None
        cuda.memcpy_htod(self.d_input, batch)

        return [self.d_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()


# For Dipoorlet
def get_dipoorlet_calib():
    data_root = f"{cfg.SYSTEM.imagenet_200_dir}/val/images/"
    image_list = get_calib_data_path()    
    for i, image_path in enumerate(image_list):
        image = Image.open(data_root + image_path).convert("RGB")
        image = Preprocess(image).numpy()
        calib_path = f"{cfg.MQBENCH.mqbench_log_dir}/dipoorlet_work_dir/input.1/"
        if not os.path.exists(calib_path):
            os.makedirs(calib_path)
        image.tofile(f"{cfg.MQBENCH.mqbench_log_dir}/dipoorlet_work_dir/input.1/" + str(i) + ".bin")


if __name__ == "__main__":
    get_dipoorlet_calib()