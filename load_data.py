import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import csv


def yield_tracking_data(path):
    '''
    A generator. Set up a for loop of the form 

    for frame, left_truth, right_truth  in read_data("my\\path\\folder"):
        # Do something

    path: the string path to the folder in which to search for videos
    yields: 
        frame: an mxnx3 numpy array of the original video frame where
            - m is the height of each video frame
            - n is the width of each frame
            - 3 is for three color channels, R, G, B (NOT BGR wich is opencv default)
        left_truth: a 1xn numpy array of the pose information for the left or only instrument
            in the frame, otherwise None
        right_truth: a 1xn numpy array of the pose information for the right instrument, 
            if any, otherwise None
    '''
    for subdir, dirs, files in os.walk(path):

        if 'Video.avi' in files:
            video_path = os.path.join(subdir, 'Video.avi')
            vid_generator = yield_video(video_path)

            left_generator = yield_none()
            right_generator = yield_none()
            if 'Pose.txt' in files:
                left_path = os.path.join(subdir, 'Pose.txt')
                left_generator = yield_pose(left_path)
            elif 'Left_Instrument_Pose.txt' in files:
                left_path = os.path.join(subdir, 'Left_Instrument_Pose.txt')
                left_generator = yield_pose(left_path)
            if 'Right_Instrument_Pose.txt' in files:
                right_path = os.path.join(subdir, 'Right_Instrument_Pose.txt')
                right_generator = yield_pose(right_path)

            for frame in vid_generator:
                yield frame, next(left_generator), next(right_generator)

def yield_segmentation_data(path):
    '''
    A generator. Set up a for loop of the form 

    for frame, left_truth, right_truth  in read_data("my\\path\\folder"):
        # Do something

    path: the string path to the folder in which to search for videos
    yields: 
        frame: an mxnx3 numpy array of the original video frame where
            - m is the height of each video frame
            - n is the width of each frame
            - 3 is for three color channels, R, G, B (NOT BGR wich is opencv default)
        left_truth: The ground truth for the left instrument if it can be found, otherwise None. If there 
            is only one instrument in the frame, it will be in left_truth. 
            The ground truth is in the form of an mxnx3 numpy array of the original video frame where
            - m is the height of each video frame
            - n is the width of each frame
            - 3 is for three color channels, R, G, B (NOT BGR wich is opencv default)
        right_truth: The ground truth for the right instrument if it can be found, otherwise None.
            The ground truth is in the form of an mxnx3 numpy array of the original video frame where
            - m is the height of each video frame
            - n is the width of each frame
            - 3 is for three color channels, R, G, B (NOT BGR wich is opencv default)
    '''
    for subdir, dirs, files in os.walk(path):

        if 'Video.avi' in files:
            video_path = os.path.join(subdir, 'Video.avi')
            vid_generator = yield_video(video_path)

            left_generator = yield_none()
            right_generator = yield_none()
            if 'Segmentation.avi' in files:
                left_path = os.path.join(subdir, 'Segmentation.avi')
                left_generator = yield_video(left_path)
            elif 'Left_Instrument_Segmentation.avi' in files:
                left_path = os.path.join(subdir, 'Left_Instrument_Segmentation.avi')
                left_generator = yield_video(left_path)
            if 'Right_Instrument_Segmentation.avi' in files:
                right_path = os.path.join(subdir, 'Right_Instrument_Segmentation.avi')
                right_generator = yield_video(right_path)

            for frame in vid_generator:
                yield frame, next(left_generator), next(right_generator)

def yield_none():
    '''
    Helper function that just yields None forever
    Don't ask
    '''
    while True:
        yield None

def yield_video(path):
    '''
    A generator. Set up a for loop of the form 

    for frame in read_data("my\\path\\video.avi"):
        # Do something

    path: the string path to the video
    yields: an mxnx3 numpy array where
        - m is the height of each video frame
        - n is the width of each frame
        - 3 is for three color channels, R, G, B
    '''
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)/255.0
            yield img
        else:
            break

def yield_pose(path):
    '''
    A generator. Set up a for loop of the form 

    for frame in read_data("my\\path\\pose.txt"):
        # Do something

    path: the string path to the txt file containing poses in csv format
    yields: an 1xn numpy array where
        - n is the number of fields in the pose description table
    '''
    pose_data = np.loadtxt(path, delimiter=' ', usecols=(0,1,2,3,4,5,6))
    for row in pose_data:
        yield row


def yield_videos(path):
    '''
    A generator. Set up a for loop of the form 

    for frame in read_data("my\\path\\folder"):
        # Do something

    path: the string path to the folder in which to search for videos
    yields: an mxnx3 numpy array where
        - m is the height of each video frame
        - n is the width of each frame
        - 3 is for three color channels, R, G, B
    '''
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1].lower() == '.avi':
                file_path = os.path.join(subdir, file)
                for frame in yield_video(file_path):
                    yield frame


def load_video(path):
    '''
    Loads a single entire video
    Parameters:
    path: the string path to the video
    returns: an fxmxnx3xp numpy array where
        - f is the total number of frames in the video
        - m is the height of each video frame
        - n is the width of each frame
        - 3 is for three color channels, R, G, B        
        The values will be floating point between 0 and 1. 
    '''
    cap = cv.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    frames = []
    while True:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)/255.0
            frames.append(img)
        else:
            break
    video = np.stack(frames, axis=0)
    
    # When everything done, release the capture
    cap.release()

    return video


if __name__ == "__main__":

    for frame, left, right in yield_tracking_data("Tracking_train"):
        print(left)
        if cv.waitKey(1) == ord('q'):
            break
    print("done :)")
    