import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
from segmentation_model import *
from segmentation_error import *

# Source data:
# https://opencas.webarchiv.kit.edu/?q=node/30


def main():

    # Part One: Segmentation

    segmentation_train_folder = "Segmentation_train"

    # TODO actually implement some training
    # for frame, left_truth, right_truth  in yield_segmentation_data(segmentation_train_folder):
    #     # TODO train the model here
    #     pass


    segmentation_test_folder = "Segmentation_test"

    errors = []
    for frame, left_truth, right_truth  in yield_segmentation_data(segmentation_test_folder):

        # TODO test the model here        
        left_guess, right_guess = model_segmentation(frame) # TODO update the model to use values from training

        #Display the resulting frame
        cv.imshow('frame', right_guess)

        if cv.waitKey(1) == ord('q'):
            break

        errors.append(get_segmentation_error(left_truth, left_guess))

        if right_truth is not None:
            errors.append(get_segmentation_error(right_truth, right_guess))


    # # Report the overall accuracy
    errors = np.vstack(errors)
    mean_error = np.mean(errors, axis=0)

    labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
    n_bins = 20

    figure = plt.figure()

    for i in range(0,4):
        print(labels[i] + " Average: %f" % mean_error[i])
        plt.subplot(2, 2, i+1)
        plt.hist(errors[:,i], bins=n_bins, range=(0,1))
        plt.xlim([0,1])
        plt.xlabel("Rate")
        plt.ylabel("Count")
        plt.title(labels[i])
        #plt.yscale('log')

    plt.show()
    
    # # TODO maybe there are some nice plots we can make, or output some of the segmented images


    # # Part 2: Tracking

    # tracking_train_folder = "Tracking_train"

    # for frame, left_truth, right_truth  in yield_tracking_data(tracking_train_folder):
    #     # TODO train the model here
    #     pass

    # tracking_test_folder = "Tracking_test"

    # errors = []
    # for frame, left_truth, right_truth  in yield_tracking_data(tracking_test_folder):

    #     # TODO test the model here        
    #     left_guess, right_guess = model_tracking(frame, fit_parameters_from_previous_part) # TODO make this model!

    #     error = (left_guess-left_truth) / left_truth

    #     if right_truth is not None:
    #         right_error = (right_guess-right_truth) / right_truth
    #         error = (error + right_error)/2

    #     errors.append(error)


    # # Average the error for each metric
    # errors = np.vstack(errors)
    # mean_error = numpy.mean(errors, axis=0)
    # print("Mean Error for the Seven Pose Variables:")
    # print(mean_error)

    # TODO: Maybe a histogram or some other visualization of errors would have been good


if __name__ == "__main__":
    main()