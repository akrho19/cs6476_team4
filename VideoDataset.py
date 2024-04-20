import os
from load_data import *
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, dir_name, transform=None, target_transform=None):
        print("VideoDataset __init__ ", dir_name)
        self.images = []
        self.poses = []
        for frame, left_truth, right_truth  in yield_tracking_data(dir_name):
            self.images.append(Image.fromarray(frame))
            if left_truth is None:
                left_truth = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            left_truth[0] = left_truth[0] / 720
            left_truth[1] = left_truth[1] / 576
            right_truth[0] = right_truth[0] /720
            right_truth[1] = right_truth[1] / 576
            self.poses.append(torch.tensor(np.concatenate((left_truth,right_truth)), dtype=torch.float))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        image = self.images[idx]
        pose = self.poses[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            pose = self.target_transform(pose)
        return image, pose