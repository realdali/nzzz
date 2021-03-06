from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="data/Image", help="path to dataset")
    parser.add_argument("--anno_path", type=str, default="data/Annotation/", help="标注路径")
    parser.add_argument("--model_def", type=str, default="config/custom.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/ckpt_89.pth", help="path to weights file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(opt.weights_path))
        else:
            model.load_state_dict(torch.load(opt.weights_path, map_location='cpu'))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.img_path, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    #classes = ['带电芯充电宝', '不带电芯充电宝']  # class_name
    classes = ['core_battery', 'coreless_battery']

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    core_battery_file = open('predicted_file/core_battery.txt', 'w')   #只写模式打开file
    coreless_battery_file = open('predicted_file/coreless_battery.txt', 'w')   #只写模式打开file
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        img = np.array(Image.open(path))

        filename = path.split("/")[-1].split(".")[0]
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # print(filename, classes[int(cls_pred)], cls_conf.item(),x1, y1, x2, y2)
                # classes[int(cls_pred)]
                line_info = "%s %.5f %.2f %.2f %.2f %.2f\n" % (filename, cls_conf.item(),x1, y1, x2, y2)
                if classes[int(cls_pred)] == 'core_battery':
                    core_battery_file.writelines(line_info)
                else:
                    coreless_battery_file.writelines(line_info)
    core_battery_file.close()
    coreless_battery_file.close()


