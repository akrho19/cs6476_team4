import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
import amber_load_data
from segmentation_error import *
from visualize_results import *

from amber_segmentation_model import *
from amber_tracking_model import *
#from nidhi_tracking_model import *
# TODO add the imports for your models here

# Source data:
# https://opencas.webarchiv.kit.edu/?q=node/30


def segmentation(): 
    segmentation_train_folder = "Segmentation_train"

    # TODO actually implement some training if you want
    # Maybe we save this for next time though

    segmentation_test_folder = "Segmentation_test"

    errors = []
    iou = []
    dice = []
    for frame, left_truth, right_truth  in amber_load_data.yield_segmentation_data(segmentation_test_folder):

        # TODO replace this line with whatever model you would like to make
        # Write your model as its own function in the file segmetnation_model.py
        # Your model must return the same things, 
        # But may take in additional parameters such as weights calculated during training 
        left_guess, right_guess =  model_segmentation_by_color(frame) # model_segmentation_by_color(frame) 
        #left_guess, right_guess =  model_segmentation_by_blobs(frame) 
        #keypoints, descriptors, segmented_img, left_guess, right_guess = model_segmentation_by_sift(frame)
        #prev_frame = frame
        #prev_keypoints = keypoints
        #prev_descriptors = descriptors
        #matches = match_features(prev_descriptors, descriptors)

        #matched_img = cv.drawMatches(prev_frame, prev_keypoints, frame, keypoints, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        

        #Display the resulting frame
        # or at least half of it
        #cv.imshow('frame', matched_img)

        #Display the resulting frame
        # or at least half of it
        cv.imshow('frame', left_guess)

        if cv.waitKey(1) == ord('q'):
            break

        #errors.append(get_segmentation_error(left_truth, left_guess))
 
        iou.append(get_intersection_over_union(left_truth, left_guess))
        dice.append(get_DICE(left_truth,left_guess))

        if right_truth is not None:
            errors.append(get_segmentation_error(right_truth, right_guess))
            iou.append(get_intersection_over_union(right_truth, right_guess))
            dice.append(get_DICE(right_truth,right_guess))

        

    # Report the overall accuracy
    #errors = np.vstack(errors)
    iou = np.array(iou).reshape(-1, 1)
    # labels = ["True Positive", "False Positive", "False Negative", "True Negative"]
    # make_histograms(errors, labels, xlim=[0,1])
    make_histograms(iou, ["IoU"], xlabel=["IoU"], ylabel=["Count"], xlim=None, n_bins=20)

    dice = np.array(dice).reshape(-1, 1)
    make_histograms(dice, ["DICE"], xlabel=["DICE"], ylabel=["Count"], xlim=None, n_bins=20)


def tracking():
    tracking_train_folder = "Tracking_train"

    # TODO train the model here

    tracking_test_folder = "Tracking_test"

    errors = []
    for frame, left_truth, right_truth  in amber_load_data.yield_tracking_data(tracking_test_folder):

        # left_truth = left_truth.tolist()
        # if right_truth is not None:
        #     right_truth = right_truth.tolist()

        # TODO test the model here        
        #left_guess, right_guess = model_tracking_by_sift(frame) # TODO make this model!
        left_guess, right_guess, shaft_left, shaft_right, head_left, head_right = model_tracking_by_color(frame)

        # print(left_shaft)
        # print(np.shape(left_shaft))

        left_error = get_tracking_error(left_truth, left_guess)
        errors.append(left_error)

        # Display annotated frame
        #annotated_frame = frame
        annotated_frame = np.stack([shaft_left, head_left, np.zeros_like(shaft_left)], 2)

        annotated_frame = visualize_pose(annotated_frame, left_guess, (255, 0, 255)) # Magenta
        annotated_frame = visualize_pose(annotated_frame, left_truth, (0, 255, 255)) # Cyan
        if right_truth is not None and right_guess is not None:
            annotated_frame = annotated_frame + np.stack([shaft_right, head_right, np.zeros_like(shaft_right)], 2)
            annotated_frame = visualize_pose(annotated_frame, right_guess, (255, 255, 0)) # Yellow
            annotated_frame = visualize_pose(annotated_frame, right_truth, (255, 255, 255)) # White

        annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR)
        cv.imshow('frame', annotated_frame)

        if cv.waitKey(1) == ord('q'):
            break

        if right_truth is not None:
            right_error = get_tracking_error(right_truth, right_guess)
            errors.append(right_error)

    # Report the overall accuracy
    errors = np.vstack(errors)
    labels = ["tracked_point_x", " tracked_point_y", "shaft_axis_angle", "head_axis_angle", "Euclidian Distance"] #, "clasper_angle"] # TODO: Uncomment if you'd like to try with clasper_angle
    make_histograms(errors, labels, xlim=None, n_bins=20, \
        ylabel = ["Frames", "Frames", "Frames", "Frames", "Frames"], \
        xlabel=["Error [Pixels]", "Error [Pixels]", "Error [Degrees]", "Error [Degrees]", "Error [Pixels]"])


def main():

    # Part One: Segmentation
    #segmentation()

    # Part 2: Tracking
    # TODO: uncomment this if you want to test tracking
    tracking()

if __name__ == "__main__":
    main()
