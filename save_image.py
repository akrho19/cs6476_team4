import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
from segmentation_error import *
from visualize_results import *
from cloud_segmentation_model import *
from amber_tracking_model import *

'''
A utility that can help you save images for use in the report. 
'''



segmentation_test_folder = "Segmentation_test\Dataset3"

for frame, left_truth, right_truth in yield_segmentation_data(segmentation_test_folder):
    print("Saving image")

    left_guess, right_guess =  model_segmentation_by_blobs(frame) 

    figure = plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(frame)
    plt.title("Image")
    plt.subplot(1,3,2)
    plt.imshow(left_truth)
    plt.title("Ground Truth")
    plt.subplot(1,3,3)
    plt.imshow(left_guess, cmap='gray')
    plt.title("Algorithm Output")
    
    plt.savefig("blob_segmentation_sigma_151.png")

    break
