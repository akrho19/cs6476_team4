'''
In this file, write the code to segment an unseen image
If there is a new model you would like to try, don't
delete what's here; just add a new function in this file
and change which function is called in main.py
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv

def model_segmentation_by_blobs(frame, ksize = 151, threshold = 250):
    '''
    Segments the image frame into left and right tools
    Parameters:
    frame: a dimensional nxmx3 uint8 numpy array, where the third dimension is RBG color
        in the range 0-255
    Returns:
    left_guess: a nxmx1 binary numpy array. True values represent the pixel locations of
        the left (or only) tool in the image. If there is no tool, all values are False
    right_guess: a nxmx1 binary numpy array. True values represent the pixel locations 
        of the right tool, if any. If there is no second tool, all values will be False.
    '''

    # This initial attempt at blob detection doesn't really segment, it just shows the unhelpful output of the blob detection filter
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    left_guess, right_guess = (np.zeros_like(gray), np.zeros_like(gray))

    sigma = 0.3*((ksize-1)*0.5-1)
    k = 2

    Gsigma = cv.GaussianBlur(frame,(ksize,ksize),sigma,sigma)
    Gksigma = cv.GaussianBlur(frame,(ksize,ksize),k*sigma,k*sigma)

    DoG = Gksigma - Gsigma

    gray_DoG = cv.cvtColor(DoG, cv.COLOR_BGR2GRAY)

    th, dst = cv.threshold(gray_DoG,threshold,255,cv.THRESH_BINARY)

    cv.imshow("Difference of Gaussians",dst)

    return dst, right_guess