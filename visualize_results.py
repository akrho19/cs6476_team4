'''
Use this file to write any functions related to display of results or data. 
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
from segmentation_error import *
import math

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

VECTOR_LENGTH = 60

def make_histograms(data, labels, xlabel="Rate", ylabel="Count", xlim=[0,1], n_bins=20):
    '''
    Makes a figure with a histogram subplot for each column in data.
    Parameters:
    data: A numpy array with n columns. n subplots will be created.
    labels: A list of n strings, the titles for each plot
    xlabel: the x axis label to be used for all subplots. Default is "Rate"
    ylabel: the y axis label to be used for all plots. Default is "Count"
    xlim: A list of length 2 with the lower and upper bound of the x axis. Default [0,1].
    n_bins: Number of histogram bins to use, default 20.
    returns: None
    '''

    figure = plt.figure()
    for i in range(0,data.shape[1]):
        print(labels[i] + " Average: %f" % np.mean(data[:,i]))
        plt.subplot(2, -(data.shape[1]//-2), i+1)
        plt.hist(data[:,i], bins=n_bins, range=(0,1))
        plt.xlim(xlim)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(labels[i])

    plt.show()


def visualize_pose(frame, pose):
    '''
    Returns a frame that, when displayed as an image
    Is the original frame with annotations showing the pose
    of the robot given by pose
    Parameters:
        frame: The original frame; a nxmx3 numpy array representing
            an RGB image on the scale 0 to 255
        pose: A 1x7 numpy array representing:
            [tracked_point_x,  tracked_point _y, shaft_axis_x, shaft_axis_y,
                head_axis_x, head_axis_y, clasper_angle]
    Returns:
        The annotated frame; a nxmx3 numpy array representing
            an RGB image on the scale 0 to 255
    '''
    tracked_point_x = pose[0]
    tracked_point_y = pose[1]
    shaft_axis_x = pose[2]
    shaft_axis_y = pose[3]
    head_axis_x = pose[4]
    head_axis_y = pose[5]
    clasper_angle = pose[6]

    # Draw the shaft axis
    start_point = (round(tracked_point_x), round(tracked_point_y))
    end_point = (round(tracked_point_x + shaft_axis_x*VECTOR_LENGTH), \
                round(tracked_point_y + shaft_axis_y*VECTOR_LENGTH))
    frame = cv.line(frame, start_point, end_point, BLUE, 5)

    # Draw the tool axis
    end_point = (round(tracked_point_x + head_axis_x*VECTOR_LENGTH), \
                round(tracked_point_y + head_axis_y*VECTOR_LENGTH))
    frame = cv.line(frame, start_point, end_point, RED, 5)

    # Draw the tracked point
    frame = cv.circle(frame, start_point, 10, YELLOW, -1)

    # Draw the angle
    axesLength = (20,20)
    angle = math.atan2(head_axis_y, head_axis_x)
    startAngle = 0
    endAngle = clasper_angle
    frame = cv.ellipse(frame, start_point, axesLength, angle, startAngle, endAngle, GREEN, 5)

    return frame