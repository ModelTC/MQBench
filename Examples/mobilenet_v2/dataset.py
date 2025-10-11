import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from common.configs import get_cfg_defaults
cfg = get_cfg_defaults()

class ImagesDataset(Dataset):
    def __init__(self, files, labels, encoder, transforms, mode):
        super().__init__()
        self.files = files
        self.labels = labels
        self.encoder = encoder
        self.transforms = transforms
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        pic = Image.open(self.files[index]).convert("RGB")

        if self.mode == "train" or self.mode == "val":
            x = self.transforms(pic)
            label = self.labels[index]
            y = self.encoder.transform([label])[0]
            return x, y
        elif self.mode == "test":
            x = self.transforms(pic)
            return x, self.files[index]


def get_dataset():
    DIR_MAIN = cfg.SYSTEM.imagenet_200_dir
    DIR_TRAIN = os.path.join(DIR_MAIN, "train/")
    DIR_VAL = os.path.join(DIR_MAIN, "val/")
    DIR_TEST = os.path.join(DIR_MAIN, "test/")

    labels = os.listdir(DIR_TRAIN)
    encoder_labels = LabelEncoder()
    encoder_labels.fit(labels)

    files_train = []
    labels_train = []
    for label in labels:
        for filename in os.listdir(DIR_TRAIN + label + "/images/"):
            files_train.append(DIR_TRAIN + label + "/images/" + filename)
            labels_train.append(label)

    files_val = []
    labels_val = []
    for filename in os.listdir(DIR_VAL + "images/"):
        files_val.append(DIR_VAL + "images/" + filename)

    val_df = pd.read_csv(
        DIR_VAL + "val_annotations.txt",
        sep="\t",
        names=["File", "Label", "X1", "Y1", "X2", "Y2"],
        usecols=["File", "Label"],
    )
    for f in files_val:
        l = val_df.loc[val_df["File"] == f[len(DIR_VAL + "images/") :]]["Label"].values[
            0
        ]
        labels_val.append(l)

    # List of files for testing (10'000 items)
    files_test = []
    for filename in os.listdir(DIR_TEST + "images/"):
        files_test.append(DIR_TEST + "images/" + filename)
        files_test = sorted(files_test)

    transforms_train = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            transforms.RandomErasing(
                p=0.5, scale=(0.06, 0.08), ratio=(1, 3), value=0, inplace=True
            ),
        ]
    )

    transforms_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ]
    )

    train_dataset = ImagesDataset(
        files=files_train,
        labels=labels_train,
        encoder=encoder_labels,
        transforms=transforms_train,
        mode="train",
    )

    val_dataset = ImagesDataset(
        files=files_val,
        labels=labels_val,
        encoder=encoder_labels,
        transforms=transforms_val,
        mode="val",
    )

    test_dataset = ImagesDataset(
        files=files_test,
        labels=None,
        encoder=None,
        transforms=transforms_val,
        mode="test",
    )

    return train_dataset, val_dataset, test_dataset


# train_dataset, val_dataset, test_dataset = get_dataset()

# print(val_dataset[0])