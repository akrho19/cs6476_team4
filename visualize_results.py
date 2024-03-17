'''
Use this file to write any functions related to display of results or data. 
'''

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
from segmentation_model import *
from segmentation_error import *

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