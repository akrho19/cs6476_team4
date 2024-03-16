import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *

# Source data:
# https://opencas.webarchiv.kit.edu/?q=node/30


def main():

    # Part One: Segmentation

    segmentation_train_folder = "Segmentation_train"

    for frame, left_truth, right_truth  in read_data(segmentation_train_folder):
        # TODO train the model here
        pass


    segmentation_test_folder = "Segmentation_test"

    accuracy = []
    for frame, left_truth, right_truth  in read_data(segmentation_test_folder):

        # TODO test the model here        
        left_guess, right_guess = model_segmentation(frame, fit_parameters_from_previous_part) # TODO make this model!

        incorrect_pixels = np.sum(numpy.logical_xor(left_truth, left_guess))

        if right_truth is not None:
            incorrect_pixels += np.sum(numpy.logical_xor(right_truth, right_guess))

        accuracy.append((frame.size - incorrect_pixels)/frame.size)

    # Report the overall accuracy
    average_accuracy = sum(xs) / len(xs)
    print("Average accuracy: %f" % average_accuracy)

    # TODO maybe there are some nice plots we can make, or output some of the segmented images


    # Part 2: Tracking

    tracking_train_folder = "Tracking_train"

    for frame, left_truth, right_truth  in read_data(tracking_train_folder):
        # TODO train the model here
        pass

    tracking_test_folder = "Tracking_test"

    errors = []
    for frame, left_truth, right_truth  in read_data(tracking_test_folder):

        # TODO test the model here        
        left_guess, right_guess = model_tracking(frame, fit_parameters_from_previous_part) # TODO make this model!

        error = (left_guess-left_truth) / left_truth

        if right_truth is not None:
            right_error = (right_guess-right_truth) / right_truth
            error = (error + right_error)/2

        errors.append(error)


    # Average the error for each metric
    errors = np.vstack(errors)
    mean_error = numpy.mean(errors, axis=0)
    print("Mean Error for the Seven Pose Variables:")
    print(mean_error)

    # TODO: Maybe a histogram or some other visualization of errors would have been good


if __name__ == "__main__":
    main()