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

def make_histograms(data, labels, xlabel=["Rate"], ylabel=["Count"], xlim=None, n_bins=20):
    '''
    Makes a figure with a histogram subplot for each column in data.
    Parameters:
    data: A numpy array with n columns. n subplots will be created.
    labels: A list of n strings, the titles for each plot
    xlabel: the x axis label to be used for all subplots. Default is "Rate"
    ylabel: the y axis label to be used for all plots. Default is "Count"
    xlim: A list of length 2 with the lower and upper bound of the x axis.
    n_bins: Number of histogram bins to use, default 20.
    returns: None
    '''

    #np.savetxt("error.csv", data, delimiter=",")

    figure = plt.figure()
    for i in range(0,len(labels)):
        currentData = data[:,i][~np.isnan(data[:,i])]
        print(labels[i] + " RMS Error: %f" % np.sqrt(np.mean(np.square(currentData))))
        print(labels[i] + " Percent Within 10 units: %f %%" % (100*(np.sum(np.absolute(currentData) <= 10)/len(currentData))))
        print(labels[i] + " Median absolute error: %f" % np.median(np.absolute(currentData)))
        plt.subplot(-(data.shape[1]//-2), 2, i+1)
        plt.hist(currentData, bins=n_bins, range=xlim)
        plt.xlim(xlim)
        plt.xlabel(xlabel[i] if len(xlabel) > 1 else xlabel[0])
        plt.ylabel(ylabel[i]if len(ylabel) > 1 else ylabel[0])
        plt.title(labels[i])

    plt.show()


def visualize_pose(frame, pose, center_hue):
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
    frame = cv.circle(frame, start_point, 10, center_hue, -1)

    # Draw the angle
    axesLength = (20,20)
    angle = math.atan2(head_axis_y, head_axis_x)
    startAngle = 0
    endAngle = clasper_angle
    frame = cv.ellipse(frame, start_point, axesLength, angle, startAngle, endAngle, GREEN, 5)

    return frame