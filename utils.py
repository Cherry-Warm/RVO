from __future__ import print_function, division

from glob import glob

import random
import cv2
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.transforms import CenterCrop, Grayscale, RandomHorizontalFlip, RandomRotation

from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

LABEL_ENUM = {0: "nrml", 1: "benign", 2: "malg"}


class SomeDataset(Dataset):
    def __init__(self, img_dir, df, labels, to_blur=True, blur_kernel_size=(1,1), sigma=0, img_transforms=None):
        self.img_dir = img_dir
        self.transforms = img_transforms
        d = []
        for label in labels:
            key, cls = label.split(",")
            val = df[key]
            val["name"] = key
            val["label"] = int(cls)
            d.append(val)
        self.df = d
        self.sigma = sigma
        self.to_blur = to_blur
        self.blur_kernel_size = blur_kernel_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Get the image
        file_name = self.df[idx]["name"]
        img_name = os.path.join(self.img_dir, file_name)
        image = cv2.imread(img_name)
        if self.to_blur:
            image = cv2.GaussianBlur(image, self.blur_kernel_size, self.sigma)
        if self.transforms:
            img = self.transforms(image)
        label = torch.as_tensor(self.df[idx]["label"], dtype=torch.int64)
        #cv2.imwrite(filename, image)
        print
        return {'imgs': img, 'labels': label, 'names': file_name}

# 数据集定义：类用于加载医学图像数据集，包括图像路径、标签等信息
# All photoes in one bag, labels and name in a file of csv
class RVODataset(Dataset):
    """ RVO classification dataset. """
    def __init__(self, img_dir, img_transform, csv_path):
        super().__init__()
        self.img_dir = img_dir
        self.transform = img_transform
        self.csv = csv_path
        df = pd.read_csv(self.csv)
        self.info = df

    def __getitem__(self, index):
        patience_info = self.info.iloc[index]
        file_name = patience_info['name']+'.jpg'
        file_path = glob(self.img_dir+'/'+file_name)[0]
        file_name = file_name.split('.')[0]
        label = patience_info['label']
        img = Image.open(file_path)
        if self.transform is not None:
            img = self.transform(img)

        return {'imgs': img, 'labels': label, 'names': file_name}

    def __len__(self):
        return len(self.info)

# 数据预处理和转换：函数定义了数据集的预处理和转换，根据训练和测试模式使用不同的数据转换。
def get_dataset(imgpath, csvpath, img_size, mode='train', keyword=None):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.CenterCrop((img_size, img_size)),
        AddGaussianNoise(amplitude=random.uniform(0, 1), p=0.5),
        AddBlur(kernel=3, p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.5, 2), contrast=(0.5, 2)),
        # transforms.RandomRotation((-20, 20)),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=0.5, std=0.5)
    ])

    if mode =='train':
        transform = train_transform
    elif mode == 'test':
        transform = test_transform

    dataset = RVODataset(imgpath, transform, csvpath)

    return dataset

# 数据增强
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            img = N + img
            img[img > 255] = 255
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img

class AddBlur(object):
    def __init__(self, kernel=3, p=1):
        self.kernel = kernel
        self.p = p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            img = cv2.blur(img, (self.kernel, self.kernel))
            img = Image.fromarray(img.astype('uint8')).convert('L')
            return img
        else:
            return img


# 混淆矩阵计算
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.flatten(preds)
    labels = torch.flatten(labels)
    for p, t in zip(preds, labels):
        conf_matrix[int(p), int(t)] += torch.tensor(1)
    return conf_matrix


def calculate_confusion_matrix(predicted_labels, true_labels, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for pred, true in zip(predicted_labels, true_labels):
        confusion_matrix[true][pred] += 1

    return confusion_matrix


def cal_sen_spec(predicted_labels, true_labels, positive_label):
    TP = sum((p == positive_label) and (t == positive_label) for p, t in zip(predicted_labels, true_labels))
    TN = sum((p != positive_label) and (t != positive_label) for p, t in zip(predicted_labels, true_labels))
    FP = sum((p == positive_label) and (t != positive_label) for p, t in zip(predicted_labels, true_labels))
    FN = sum((p != positive_label) and (t == positive_label) for p, t in zip(predicted_labels, true_labels))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return sensitivity, specificity

def cal_TPNF(conf_matrix):
    TP = conf_matrix[1:, 1:].sum()  # True Positive
    FN = conf_matrix[1:, 0].sum()    # False Negative
    TN = conf_matrix[0, 0]           # True Negative
    FP = conf_matrix[0, 1:].sum()    # False Positive

    return TP, FN, TN, FP
