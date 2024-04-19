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
from PIL import Image
from sklearn.preprocessing import StandardScaler
import glob

def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:

    paths = glob.glob(dir_name+"/*/*/*.avi")
    scaler = StandardScaler()
    for path in paths:
        pixels = np.reshape(np.array(list(Image.open(path).convert(mode="L").getdata())),(-1,1))
        normalized_pixels = np.divide(pixels,255.0)
        scaler.partial_fit(normalized_pixels)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    return mean, std

def model_tracking(frame):

    ksize = 3



    return [1,1,1,1,1,1,1], [1,1,1,1,1,1,1]