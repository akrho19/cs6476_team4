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



def model_segmentation_by_sift(frame):
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
    # Note - probably DON'T resize the image
    # Unless you upscale it again at the end
    # Because the output needs to be the same size as the input
  
    
        # Blur only red 

    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    low_H = 215/2
    high_H = 330/2
    low_S = 0*2.55
    high_S = 90*2.55
    low_V = 0*2.55
    high_V = 90*2.55
    lower_bound = np.array([215/2, 0*2.55, 0*2.55])  # Lower bound for green in HSV
    upper_bound = np.array([330/2, 90*2.55, 90*2.55])  # Upper bound for green in HSV

    #lower_bound = np.array([100, 50, 50])  # Lower bound for blue in HSV
    #upper_bound = np.array([130, 255, 255])

    # Create a mask based on the color threshold
    mask = cv.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the original image
    masked_image = cv.bitwise_and(frame, frame, mask=mask)

    # Apply a blur operation to the masked image
    blurred_image = cv.GaussianBlur(masked_image, (51, 51), 0)

    # Invert the mask
    mask = cv.bitwise_not(mask)

    # Apply the inverted mask to the original image
    masked_background = cv.bitwise_and(frame, frame, mask=mask)

    # Combine the original image and the blurred image
    frame = cv.add(masked_background, blurred_image)
    
    # Sharpen image tool points
    
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    frame = cv.filter2D(frame, -1, kernel)
    
    gamma=0.9
    frame1=np.power(frame,gamma)
    if frame1.dtype != 'uint8':
        frame1 = cv.convertScaleAbs(frame1)
    frame1 = cv.Canny(frame1, 600, 870)
    kernel_size = 21
    frame = cv.GaussianBlur(frame,(kernel_size,kernel_size),0)
    
    # Edges are weird. Get them out
    frame[0:15,:] = 0
    frame[-15:-1,:] = 0
    frame[:,0:15] = 0
    frame[:,-15:-1] = 0
    # Dilate and erode
    
    kernel = np.ones((3, 3), np.uint8)
    frame = cv.erode(frame, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    frame = cv.dilate(frame, kernel, iterations=2)  
    gray_frame =  cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
    

    # Initialize the SIFT detector
    sift = cv.SIFT_create()

    # Detect keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_frame, None)

    # Create a mask to store the segmented regions
    mask = np.zeros_like(gray_frame)

    # Draw keypoints on the mask (regions of interest)
    for kp in keypoints:
        x, y = np.int32(kp.pt)
        size = np.int32(kp.size)
        cv.circle(mask, (x, y), 19, 255, -1)        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        segmented_img = cv.bitwise_and(frame, frame, mask=mask)

    frame_with_keypoints = cv.drawKeypoints(frame, keypoints, None)

    cv.imshow('Video with SIFT Keypoints', frame_with_keypoints)
    frame = gray_frame

    # Keep the two biggest curves
    gray_seg =  cv.cvtColor(segmented_img, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(gray_seg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )

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

