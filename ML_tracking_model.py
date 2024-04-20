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
import torch.optim as optim
from torchvision.io import read_video
from PIL import Image
from sklearn.preprocessing import StandardScaler
from ML_tracking_net import TrackNet
from ML_utils import *
from trainer import Trainer
from load_data import *
import time
#import glob

def compute_mean_and_std(dir_name: str):
    print("ML_tracking_model compute_meand_and_std")
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

def get_fundamental_transforms(inp_size: tuple, pixel_mean: np.array, pixel_std: np.array):
    print("ML_tracking_model get_fundamental_transforms")
    return transforms.Compose(
        [
            transforms.Resize(size=inp_size,antialias=True),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            #transforms.Normalize(pixel_mean,pixel_std)
        ]
    )

def get_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Returns the optimizer initializer according to the config on the model.

    Args:
    - model: the model to optimize for
    - config: a dictionary containing parameters for the config
    Returns:
    - optimizer: the optimizer
    """
    print("ML_tracking_model get_optimizer")
    optimizer_type = config["optimizer_type"]
    learning_rate = config["lr"]
    weight_decay = config["weight_decay"]

    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    return optimizer

def training():
    print("ML_tracking_model training")
    mean, std = compute_mean_and_std("Tracking_train")
    model = TrackNet()
    optimizer_config = {"optimizer_type":"adam", "lr":1.0e-3, "weight_decay":6e-4}
    optimizer = get_optimizer(model, optimizer_config)
    fundamental_transforms = get_fundamental_transforms((576,720),mean,std)
    model_base_path = 'model_checkpoints\\'

    trainer_instance = Trainer(train_dir="Tracking_train",
                                  test_dir="Tracking_test",
                                  model = model,
                                  optimizer = optimizer,
                                  model_dir = os.path.join(model_base_path, 'tracking_net'),
                                  train_data_transforms = fundamental_transforms,
                                  test_data_transforms = fundamental_transforms,
                                  batch_size = 30,
                                  load_from_disk = False,
                                  cuda = True,
                                 )
    start = time.time()
    trainer_instance.train(num_epochs=30)
    end = time.time()
    print("The training time taken for the learning-based tracking model is {:.9f}".format(end-start))
    return trainer_instance


def model_tracking_by_ML(frame):

    model = TrackNet()
    model.load_state_dict(torch.load('model_checkpoints\\tracking_net\\checkpoint.pt'))
    model.eval()

    pose = model(get_fundamental_transforms()(Image.fromarray(frame))).tolist()
    pose[0] = pose[0]* 720
    pose[1] = pose[1]* 576
    pose[7] = pose[7]* 720
    pose[8] = pose[8]* 576
    return pose[0:7], pose[7:14]

if __name__ == "__main__":
    print("ML_tracking_model main")
    trainer_instance = training()
    #trainer_instance.plot_accuracy()
    trainer_instance.plot_loss_history