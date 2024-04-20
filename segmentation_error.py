import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
import math



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

def get_DICE(truth,guess):
    truth = np.any(truth, axis=2)

    union = np.logical_or(truth, guess)
    intersection = np.logical_and(truth, guess)

    return 2*np.sum(intersection)/(np.sum(truth)+np.sum(guess))

def get_tracking_error(truth, guess):
    truth_tracked_point_x = truth[0]
    truth_tracked_point_y = truth[1]
    truth_shaft = np.array([truth[2], truth[3]])
    truth_head = np.array([truth[4], truth[5]])
    truth_clasper_angle = truth[6]

    guess_tracked_point_x = guess[0]
    guess_tracked_point_y = guess[1]
    guess_shaft = np.array([guess[2], guess[3]])
    guess_head = np.array([guess[4], guess[5]])
    guess_clasper_angle = guess[6]

    dx = guess_tracked_point_x - truth_tracked_point_x
    dy = guess_tracked_point_y - truth_tracked_point_y
    d_euclidean = (dx**2 + dy**2) **.5

    try:
        d_shaft_angle = math.acos(np.dot(truth_shaft, guess_shaft)/(np.linalg.norm(truth_shaft)*np.linalg.norm(guess_shaft)))*180/math.pi
    except:
        d_shaft_angle = 0

    try:
        d_head_angle = math.acos(np.dot(truth_head, guess_head)/(np.linalg.norm(truth_head)*np.linalg.norm(guess_head)))*180/math.pi
    except:
        d_head_angle = 0
        
    d_clasper = truth_clasper_angle - guess_clasper_angle

    return dx, dy, d_shaft_angle, d_head_angle, d_euclidean, d_clasper

