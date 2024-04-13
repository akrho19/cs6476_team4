'''
In this file, write the code to track the endoscopic 
tools in an unseen image
If there is a new model you would like to try, don't
delete what's here; just add a new function in this file
and change which function is called in main.py
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv
import math
from amber_segmentation_model import *
from nidhi_segmentation_model import *

# ["tracked_point_x", " tracked_point _y", "shaft_axis_x", "shaft_axis_y", \
# "head_axis_x", "head_axis_y", "clasper_angle"]

def model_tracking_by_color(frame):
    color_left, color_right =  model_segmentation_by_color(frame) 
    head_left, head_right = head_segmentation_by_color(frame)

    color_left = color_left.astype(np.uint8)
    head_left = head_left.astype(np.uint8)

    # Separate 
    left_tip, left_shaft = get_tip_and_shaft(color_left, head_left)

    # Find orientation of shaft
    shaft_axis_x_left, shaft_axis_y_left, shaft_x, shaft_y = get_shaft_orientation(left_shaft)

    # Find orientation of head
    head_axis_x_left, head_axis_y_left, head_x, head_y = get_shaft_orientation(left_tip)

    # Find center of intersection between shaft and head
    tracked_point_x_left, tracked_point_y_left = get_tracked_point(left_tip, left_shaft)
    # tracked_point_x_left, tracked_point_y_left = get_intersection_point(shaft_axis_x_left, \
    #     shaft_axis_y_left, shaft_x, shaft_y, head_axis_x_left, head_axis_y_left, head_x, head_y)

    if color_right is not None and head_right is not None:
        # Separate 
        right_tip, right_shaft = get_tip_and_shaft(color_right, head_right)

        # Find orientation of shaft
        shaft_axis_x_right, shaft_axis_y_right, shaft_x, shaft_y = get_shaft_orientation(right_shaft)

        # Find orientation of head
        head_axis_x_right, head_axis_y_right, head_x, head_y = get_shaft_orientation(right_tip)

        # Find center of intersection between shaft and head
        tracked_point_x_right, tracked_point_y_right = get_tracked_point(right_tip, right_shaft)
        # tracked_point_x_right, tracked_point_y_right = get_intersection_point(shaft_axis_x_right, \
        #     shaft_axis_y_right, shaft_x, shaft_y, head_axis_x_right, head_axis_y_right, head_x, head_y)

        return [tracked_point_x_left, tracked_point_y_left, shaft_axis_x_left, shaft_axis_y_left, head_axis_x_left, head_axis_y_left,0], \
            [tracked_point_x_right, tracked_point_y_right, shaft_axis_x_right, shaft_axis_y_right, head_axis_x_right, head_axis_y_right,0], \
           

    return [tracked_point_x_left, tracked_point_y_left, shaft_axis_x_left, shaft_axis_y_left,head_axis_x_left, head_axis_y_left,0], \
            None

def get_tip_and_shaft(color_segment, head_segment):

    tip = np.logical_and(color_segment, head_segment).astype(np.uint8)
    
    tip, _ = get_largest_blobs(tip, second=False)
    shaft = color_segment.astype(np.uint8) - tip
    shaft, _ = get_largest_blobs(shaft, second=False)
    return tip, shaft

def get_shaft_orientation(frame):
    contours,hierarchy = cv.findContours(frame,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )
    if not contours:
        return 0,0,0,0
    cnt = contours[0]
    [vx,vy,x,y] = cv.fitLine(cnt, cv.DIST_L2,0,0.01,0.01)

    vx = vx / math.sqrt(vx*vx + vy*vy);
    vy = vy / math.sqrt(vx*vx + vy*vy);

    return vx[0], vy[0], x[0], y[0]

def get_intersection_point(vx1,vy1,x1,y1,vx2,vy2,x2,y2):
    x = 0
    y = 0
    if (vy1*vx2-vy2*vx1) !=0:
        x = ((vx2*vx1)*(y2-y1) - (x2-x1))/(vy1*vx2-vy2*vx1)
    if vx1 != 0:
        y = (vy1/vx1)*(x - x1) + y1
    elif vx2 != 0:
        y = (vy2/vx2)*(x - x2) + y2
    return x, y

def get_centroid(frame):
    mass_x, mass_y = np.where(frame)
    cx = np.average(mass_x)
    cy = np.average(mass_y)
    return cx, cy

def get_tracked_point(head, shaft):
    # dilate the shaft 
    kernel = np.ones((2, 2), np.uint8)
    shaft = cv.dilate(shaft, kernel, iterations=1) 

    # find the collisions
    border = np.logical_and(head, shaft).astype(np.uint8)

    cx, cy = get_centroid(border)

    if math.isnan(cx) or math.isnan(cy):
        return 0, 0

    # contours,hierarchy = cv.findContours(border,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE )
    # cnt = contours[0]
    # M = cv.moments(cnt)

    # cx = int(M['m10']/M['m00'])
    # cy = int(M['m01']/M['m00'])

    return cx, cy