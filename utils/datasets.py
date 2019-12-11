import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class_names = ['带电芯充电宝', '不带电芯充电宝']

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

# 计算中心点，归一化
def get_format_center(min_value, max_value, total_size):
    return (int(min_value) + int(max_value)) / 2 / total_size
# 计算距离，归一化

def get_format_size(min_value, max_value, total_size):
    return (int(max_value) - int(min_value)) / total_size


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, valid_path, images_path, labels_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        # 读取的文件为图片相对地址，为test的目录，里边是图片名称
        with open(valid_path, "r") as file:
            self.img_names = file.readlines()

        # 获取标识文件
        # TODO @yuan.zhang, 需确定路径是否加/
        self.label_files = [
            labels_path + img_name + '.txt'
            for img_name in self.img_names
        ]
        self.images_path = images_path
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_name = self.img_names[index % len(self.img_names)].rstrip()

        # Extract image as PyTorch tensor
        imgObj = Image.open(self.images_path + img_name + '.jpg')
        img = transforms.ToTensor()(imgObj.convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_names)].rstrip().replace('\n', '')

        targets = None
        if os.path.exists(label_path):
            # 数据格式转换
            # 01013328.TIFF 带电芯充电宝 631 700 737 879
            # =>
            # 0 0.515 0.5 0.21694873 0.18286777
            width, height = imgObj.size
            with open(label_path, 'r') as f:
                lines = f.readlines()
            if any(lines) == 1:
                splitlines = [x.strip().split(' ') for x in lines]
                boxes = []
                for x in splitlines:
                    if x[1] in class_names:
                        boxes.append([class_names.index(x[1]), get_format_center(x[2], x[4], width), get_format_center(x[3], x[4], height), get_format_size(x[2], x[4], width), get_format_size(x[3], x[5], height)])   
                
            boxes = torch.from_numpy(np.array(boxes))

            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_name, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_names)
