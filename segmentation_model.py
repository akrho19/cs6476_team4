import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv

# Note: The ranges for HSV in opencv
#(0–180, 0–255, 0–255)

# Might be useful: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

def model_segmentation(frame):
    # Note - probably DON'T resize the image
    # Unless you upscale it again at the end
    # Because the output needs to be the same size as the input

    # Apply gaussian blur
    kernel_size = 11
    frame = cv.GaussianBlur(frame,(kernel_size,kernel_size),0)


    # Convert to HSV
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # # # Threshold based on red
    low_H = 215/2
    high_H = 330/2
    low_S = 0*2.55
    high_S = 90*2.55
    low_V = 0*2.55
    high_V = 90*2.55
    frame = cv.bitwise_not(cv.inRange(frame, (low_H, low_S, low_V), (high_H, high_S, high_V)))

    # Edges are weird. Get them out
    frame[0:15,:] = 0
    frame[-15:-1,:] = 0
    frame[:,0:15] = 0
    frame[:,-15:-1] = 0


    # Dilate and erode
    kernel = np.ones((2,2), np.uint8)
    frame = cv.erode(frame, kernel, iterations=1) 
    kernel = np.ones((5, 5), np.uint8)
    frame = cv.dilate(frame, kernel, iterations=1) 
    kernel = np.ones((3, 3), np.uint8)
    frame = cv.erode(frame, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    frame = cv.dilate(frame, kernel, iterations=2)  
    # kernel = np.ones((5, 5), np.uint8)
    # frame = cv.dilate(frame, kernel, iterations=1) 

    # Keep the two biggest curves
    contours, hierarchy = cv.findContours(frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )

    min_contour = 2000
    maxContour = 0
    secondContour = 0
    maxContourData = None
    secondContourData = None
    for contour in contours:
        contourSize = cv.contourArea(contour)
        if contourSize > secondContour and contourSize > min_contour:
            if contourSize > maxContour:
                secondContour = maxContour
                secondContourData = maxContourData
                maxContour = contourSize
                maxContourData = contour
            else:
                secondContour = contourSize
                secondContourData = contour
    
    # Create a mask from the largest contours
    left_guess = np.zeros(frame.shape, np.uint8)
    right_guess = np.zeros(frame.shape, np.uint8)
    if secondContour > 0:
        # Find which is right and left
        M_first = cv.moments(maxContourData)
        cX_first = int(M_first["m10"] / M_first["m00"])

        M_second = cv.moments(secondContourData)
        cX_second = int(M_second["m10"] / M_second["m00"])

        if cX_second > cX_first: # first is left
            cv.drawContours(left_guess, [maxContourData], -1, 255, cv.FILLED)
            cv.drawContours(right_guess, [secondContourData], -1, 255, cv.FILLED)
        else: # first is right
            cv.drawContours(left_guess, [secondContourData], -1, 255, cv.FILLED)
            cv.drawContours(right_guess, [maxContourData], -1, 255, cv.FILLED)
    elif maxContour > 0:
        cv.drawContours(left_guess, [maxContourData], -1, 255, cv.FILLED)

    return left_guess, right_guess
