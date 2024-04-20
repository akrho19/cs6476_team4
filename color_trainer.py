import numpy as np
import cv2 as cv
import os
import csv
from load_data import *
from segmentation_error import *
from scipy.optimize import least_squares
from amber_segmentation_model_trainable import *

def objective_function(params):
    '''
    Returns the average IoU for a given set of color parameters
    '''
    segmentation_train_folder = "Segmentation_train"

    #errors = []
    iou = []
    for frame, left_truth, right_truth  in yield_segmentation_data(segmentation_train_folder):

        left_guess, right_guess =  model_segmentation_by_color_trainable(frame, params) 
        cv.imshow('frame', left_guess)
        if cv.waitKey(1) == ord('q'):
            break

        #errors.append(get_segmentation_error(left_truth, left_guess))
        iou.append(get_intersection_over_union(left_truth, left_guess))

        if right_truth is not None:
            #errors.append(get_segmentation_error(right_truth, right_guess))
            iou.append(get_intersection_over_union(right_truth, right_guess))


    # Report the overall accuracy
    #errors = np.vstack(errors)
    iou = np.array(iou).reshape(-1, 1)
    return np.mean(iou)


def optimize(init_params = [215/2,330/2,0*2.55,90*2.55,0*2.55,90*2.55]):
    '''
    Uses least squares optimization to find HSV color parameters that minize IoU
    '''
    optimized_params = least_squares(objective_function, init_params, method='trf', verbose=2, max_nfev=50000, gtol=1e-12, bounds=([0,0,0,0,0,0],[180,180,255,255,255,255])).x
    print(optimized_params)

    return optimized_params

def main():
    optimize()

if __name__ == "__main__":
    main()