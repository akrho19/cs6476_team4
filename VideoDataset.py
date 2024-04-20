import os
from load_data import *
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, dir_name, transform=None, target_transform=None):
        print("VideoDataset __init__ ", dir_name)
        frames = []
        truths = []
        for frame, left_truth, right_truth  in yield_tracking_data(dir_name):
            frames.append(Image.fromarray(frame))
            if left_truth is None:
                left_truth = np.array([0,0,0,0,0,0,0])
            truths.append(np.concatenate((left_truth,right_truth)))
        
        self.images = np.stack(frames, axis=0)
        self.poses = np.stack(truths, axis=0)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            pose = self.target_transform(pose)
        return image, pose