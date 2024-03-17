import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
from segmentation_model import *
from segmentation_error import *
from visualize_results import *
from tracking_model import *

# Source data:
# https://opencas.webarchiv.kit.edu/?q=node/30


def main():

    # Part One: Segmentation

    segmentation_train_folder = "Segmentation_train"

    # TODO actually implement some training

    segmentation_test_folder = "Segmentation_test/Dataset3"

    errors = []
    for frame, left_truth, right_truth  in yield_segmentation_data(segmentation_test_folder):

        # TODO replace this line with whatever model you would like to make
        # Write your model as its own function in the file segmetnation_model.py
        # Your model must return the same things, 
        # But may take in additional parameters such as weights calculated during training 
        left_guess, right_guess =  model_segmentation_by_color(frame) 

        #Display the resulting frame
        cv.imshow('frame', right_guess)

        if cv.waitKey(1) == ord('q'):
            break

        errors.append(get_segmentation_error(left_truth, left_guess))

        if right_truth is not None:
            errors.append(get_segmentation_error(right_truth, right_guess))


    # Report the overall accuracy
    errors = np.vstack(errors)
    labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
    make_histograms(errors, labels)


    # Part 2: Tracking

    tracking_train_folder = "Tracking_train"

    # TODO train the model here

    tracking_test_folder = "Tracking_test"

    errors = []
    for frame, left_truth, right_truth  in yield_tracking_data(tracking_test_folder):

        # TODO test the model here        
        left_guess, right_guess = model_tracking_by_color(frame) # TODO make this model!

        left_error = (left_guess-left_truth) / left_truth
        errors.append(lefterror)

        if right_truth is not None:
            right_error = (right_guess-right_truth) / right_truth
            errors.append(right_error)

    # Report the overall accuracy
    errors = np.vstack(errors)
    labels = ["I'm", "Not", "Actually", "Sure", "What", "These", "Represent"] # TODO figure out what the pose data actually is lol
    make_histograms(errors, labels)

if __name__ == "__main__":
    main()