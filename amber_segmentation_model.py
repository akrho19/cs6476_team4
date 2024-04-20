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

def model_segmentation_by_color(frame):
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

    # Apply gaussian blur
    kernel_size = 11
    frame = cv.GaussianBlur(frame,(kernel_size,kernel_size),0)


    frame = cv.bitwise_not(is_red(frame))

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
    # Might be useful: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

    return get_largest_blobs(frame, second=True)

def is_red(frame):
    # Convert to HSV
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Threshold based on red
    # Note: The ranges for HSV in opencv
    # (0–180, 0–255, 0–255)
    low_H = 215/2
    high_H = 340/2
    low_S = 0*2.55
    high_S = 90*2.55
    low_V = 0*2.55
    high_V = 90*2.55
    out = cv.inRange(frame, (low_H, low_S, low_V), (high_H, high_S, high_V))
    return out

def head_segmentation_by_color(frame):
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

    # Apply gaussian blur
    greyframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kernel_size = 25
    filtered = cv.GaussianBlur(greyframe,(kernel_size,kernel_size),0)

    red = is_red(frame)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
    red = cv.dilate(red, kernel, iterations=1) 
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
    red = cv.erode(red, kernel, iterations=1) 

    ret, black = cv.threshold(greyframe,80,255,cv.THRESH_BINARY_INV)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(30,30))
    black = cv.erode(black, kernel, iterations=1)

    unlikely_area = cv.bitwise_or(red, black)


    frame = filtered*unlikely_area + greyframe*(cv.bitwise_not(unlikely_area))

    #edges = frame

    # kernel_size = 13
    # frame = cv.GaussianBlur(frame,(kernel_size,kernel_size),0)

    # #edges = filtered
    edges = cv.Canny(frame,10,110)

    # # # Edges are weird. Get them out
    edges[0:30,:] = 0
    edges[-30:-1,:] = 0
    edges[:,0:30] = 0
    edges[:,-30:-1] = 0

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
    edges = cv.dilate(edges, kernel, iterations=1) 
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
    # edges = cv.erode(edges, kernel, iterations=1) 
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(10,10))
    # edges = cv.dilate(edges, kernel, iterations=1) 

    edges = remove_small_blobs(edges, 2000)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
    edges = cv.dilate(edges, kernel, iterations=1) 

    out = cv.bitwise_and(edges, cv.bitwise_not(unlikely_area))

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    # edges = cv.erode(edges, kernel, iterations=1) 
    # # kernel = np.ones((4, 4), np.uint8)
    # # edges = cv.dilate(edges, kernel, iterations=1) 
    # # kernel = np.ones((3, 3), np.uint8)
    # # edges = cv.erode(edges, kernel, iterations=1)
    # # kernel = np.ones((5, 5), np.uint8)
    # # edges = cv.dilate(edges, kernel, iterations=2)  
    # # # kernel = np.ones((5, 5), np.uint8)
    # # # edges = cv.dilate(edges, kernel, iterations=1) 

    # # # Edges are weird. Get them out
    # # frame[0:15,:] = 0
    # # frame[-15:-1,:] = 0
    # # frame[:,0:15] = 0
    # # frame[:,-15:-1] = 0


    # # # Dilate and erode
    # # kernel = np.ones((2,2), np.uint8)
    # # frame = cv.erode(frame, kernel, iterations=1) 
    # # kernel = np.ones((5, 5), np.uint8)
    # # frame = cv.dilate(frame, kernel, iterations=1) 
    # # kernel = np.ones((3, 3), np.uint8)
    # # frame = cv.erode(frame, kernel, iterations=1)
    # # kernel = np.ones((5, 5), np.uint8)
    # # frame = cv.dilate(frame, kernel, iterations=2)  
    # # # kernel = np.ones((5, 5), np.uint8)
    # # # frame = cv.dilate(frame, kernel, iterations=1) 
    # # # Might be useful: https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html

    return get_largest_blobs(out, second=True)

def remove_small_blobs(im, thresh):
    # Source: https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv

    im_8=im.astype(np.uint8) 
    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(im_8)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
    sizes = sizes[1:]
    nb_blobs -= 1

    # output image with only the kept components
    im_result = np.zeros_like(im)
    # for every component in the image, keep it only if it's above thresh
    for blob in range(nb_blobs):
        if sizes[blob] >= thresh:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255

    return im_result



    # out = np.zeros(frame.shape, np.uint8)
    # # Keep the two biggest curves
    # contours, hierarchy = cv.findContours(frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    # for contour in contours:
    #     contourSize = cv.contourArea(contour)
    #     if contourSize >=thresh:
    #         cv.drawContours(out, contour, -1, 255, cv.FILLED)
    # return out

def get_largest_blobs(frame, second=True):
    '''
    Helper function that returns the largest one or 2 blobs in image
    '''

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

    left_guess = np.zeros(frame.shape, np.uint8)
    right_guess = np.zeros(frame.shape, np.uint8)
    if secondContour > 0 and second:
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