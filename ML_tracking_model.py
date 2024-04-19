'''
In this file, write the code to track the endoscopic 
tools in an unseen image
If there is a new model you would like to try, don't
delete what's here; just add a new function in this file
and change which function is called in main.py
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_video
#from PIL import Image
from sklearn.preprocessing import StandardScaler
from ML_tracking_net import TrackNet
#import glob

def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    scaler = StandardScaler()
    for frame, _, _ in yield_segmentation_data(dir_name):
    #paths = glob.glob(dir_name+"/*/*/Video.avi")
    #for path in paths:
        pixels = np.reshape(np.array(frame),(-1,1))#list(Image.open(path).convert(mode="L").getdata())),(-1,1))
        normalized_pixels = np.divide(pixels,255.0)
        scaler.partial_fit(normalized_pixels)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    return mean, std

def compute_loss():

    return

def fundamental_transforms(inp_size: Tuple[int, int], pixel_mean: np.array, pixel_std: np.array) -> transforms.Compose:
        return transforms.Compose(
        [
            transforms.Resize(size=inp_size,antialias=True),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            #transforms.Normalize(pixel_mean,pixel_std)
        ]
    )

def model_tracking_by_ML(frame):

    model = TrackNet()



    return [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]