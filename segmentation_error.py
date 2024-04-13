import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *



def get_segmentation_error(truth, guess):
    truth = np.any(truth, axis=2)

    positive_count = np.sum(truth)
    negative_count = truth.size - positive_count

    true_positive = np.sum(np.logical_and(truth, guess))/positive_count
    false_positive = np.sum(np.logical_and(np.logical_not(truth), guess))/negative_count
    true_negative = np.sum(np.logical_not(np.logical_or(truth, guess)))/negative_count    
    false_negative= np.sum(np.logical_and(truth, np.logical_not(guess)))/positive_count

    return [true_positive, false_positive, false_negative, true_negative]


def get_intersection_over_union(truth, guess):
    truth = np.any(truth, axis=2)

    union = np.logical_or(truth, guess)
    intersection = np.logical_and(truth, guess)

    return np.sum(intersection)/np.sum(union)

