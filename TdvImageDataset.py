# TdvImageDataset.py

import torch
from torch.utils.data import Dataset

import os
import cv2
import numpy as np

class TdvImageDataset(Dataset):

    def __init__(self, frameFileLocs, maskFileLocs):
        assert len(frameFileLocs) == len(maskFileLocs)

        self.frameFileLocs = frameFileLocs
        self.maskFileLocs = maskFileLocs
    # end function

    def __len__(self):
        return len(self.frameFileLocs)
    # end function

    def __getitem__(self, idx):
        frameFileLoc = self.frameFileLocs[idx]
        maskFileLoc = self.maskFileLocs[idx]

        # frameFileLoc will be like:
        # /path/to/file/4e43a6eb918391418b480ce0d2f0f8679512ec7f52c8bcae82f65662269c72fb_frame.png
        # we need to get just the frame id
        frameFileName = os.path.basename(frameFileLoc)
        frameId = frameFileName.replace('_frame.png', '')

        # read frame image in as 3-channel uint8
        frameImage = cv2.imread(frameFileLoc, cv2.IMREAD_COLOR)

        # convert frameImage to a float32, and normalize pixel values to 0.0 to 1.0 range
        frameImage = frameImage.astype(np.float32) / 255.0

        # read mask image in as 1-channel uint8
        maskImage = cv2.imread(maskFileLoc, cv2.IMREAD_GRAYSCALE)

        # mask images are written to file with cars drawn as pixel value 255 so they are visible, but network needs
        # pixel value to be exactly 1 for cars, so change every pixel value > 0 to 1
        maskImage[maskImage > 0] = 1

        # convert to int64, necessary for passing into PyTorch cross entropy function
        maskImage = maskImage.astype(np.int64)

        # convert both the frame image and the mask image to PyTorch images
        # change order of frame image from height, width, numChannels to numChannels, height, width
        frameImage = torch.from_numpy(frameImage.transpose(2, 0, 1))
        maskImage = torch.from_numpy(maskImage)

        return frameImage, maskImage, frameId
    # end function

# end class


