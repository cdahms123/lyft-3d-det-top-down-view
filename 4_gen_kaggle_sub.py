# 2_test.py

from TdvImageDataset import TdvImageDataset
from UNet import UNet
import Utils

import lyft_dataset_sdk
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D

import torch
import torch.utils.data
import torch.nn.functional

import pyquaternion
import cv2
import numpy as np
import os
import glob
import collections
import natsort
import time
from tqdm import tqdm
from typing import List
from termcolor import colored
import pprint

VIS_TEST_DIR = 'vis_test'
LYFT_TEST_DATASET_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'test')

# TDV = Top Down View
TDV_TEST_IMAGES_LOC = 'tdv_test_images'

GRAPHS_LOC = 'graphs'

CSV_LOC = os.path.join(os.getcwd(), 'csv')

SAMPLE_SUB_FILE_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'sample_submission.csv')
KAGGLE_SUB_FILE_NAME = 'kaggle_sub.csv'
TEST_BATCH_SIZE = 8

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    # time bookkeeping
    startTime = time.time()

    # make the directory for saving test visualization images to, if it does not exist already
    os.makedirs(VIS_TEST_DIR, exist_ok=True)

    # load test data
    print('\n' + 'loading test data . . . ' + '\n')
    level5data = LyftDataset(data_path=LYFT_TEST_DATASET_LOC, json_path=os.path.join(LYFT_TEST_DATASET_LOC, 'data'))

    classes = ['car']

    # level5data.scene is a list (180 elements long) of dictionaries
    print('\n' + 'type(level5data.scene) = ' + str(type(level5data.scene)))  # class 'list'
    print('len(level5data.scene) = ' + str(len(level5data.scene)))  # 218 elements long

    # a single 'scene' is a 25-45 second snippet of a car's journey

    # keys for each scene dictionary are description, first_sample_token, last_sample_token, log_token, name, nbr_samples, token
    print('\n' + 'type(level5data.scene[0]) = ' + str(type(level5data.scene[0])))
    print('level5data.scene[0]: ')
    pprint.pprint(level5data.scene[0])
    print('')

    testFrameIds = []
    # for each scene . . .
    for scene in level5data.scene:
        frameId = scene['first_sample_token']
        # loop through all the frame IDs in the current scene and add them to the validation frame IDs list
        while frameId is not None and frameId != '':
            testFrameIds.append(frameId)
            # advance frame ID to the next frame (will be None or an empty string if there are no more frames in this trip)
            sample = level5data.get('sample', frameId)
            frameId = sample['next']
        # end while
    # end for

    tdvImageShape = (500, 500, 3)
    lidarRange = (200.0, 200.0, 4.5)
    zOffset = -2.0

    ### write test TDV images #######################################

    if os.path.isdir(TDV_TEST_IMAGES_LOC):
        print('\n' + 'TDV directories already exists, skipping over making TDV images . . . ' + '\n')
    else:
        # if the TDV test image directory does not exist, create it
        os.makedirs(TDV_TEST_IMAGES_LOC, exist_ok=True)

        print('\n' + 'making test TDV images . . . ' + '\n')
        for frameId in tqdm(testFrameIds):
            frameImage, maskImage = Utils.makeTdvImagesForFrame(level5data, frameId, classes, tdvImageShape, lidarRange, zOffset)
            if frameImage is not None and maskImage is not None:
                cv2.imwrite(os.path.join(TDV_TEST_IMAGES_LOC, str(frameId) + '_frame.png'), frameImage)

                # ToDo: should update this to not save blank mask images, will need to update TdvImageDataset.py
                #       to handle mask image file not being available for test as well

                cv2.imwrite(os.path.join(TDV_TEST_IMAGES_LOC, str(frameId) + '_mask.png'), maskImage)
            # end if
        # end for
    # end if

    # get file locs of test frame and mask images
    testFrameFileLocs = sorted(glob.glob(os.path.join(TDV_TEST_IMAGES_LOC, '*_frame.png')))
    print('len(testFrameFileLocs) = ' + str(len(testFrameFileLocs)))
    testMaskFileLocs = sorted(glob.glob(os.path.join(TDV_TEST_IMAGES_LOC, '*_mask.png')))
    print('len(testMaskFileLocs) = ' + str(len(testMaskFileLocs)))

    testDataset = TdvImageDataset(testFrameFileLocs, testMaskFileLocs)
    print('\n' + 'len(testDataset) = ' + str(len(testDataset)))

    testDataLoader = torch.utils.data.DataLoader(testDataset, TEST_BATCH_SIZE, shuffle=False)
    print('len(testDataLoader) = ' + str(len(testDataLoader)))

    # get device (cuda or cpu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(colored('using GPU', 'green'))
    else:
        device = torch.device('cpu')
        print(colored('CUDA does not seem to be available, using CPU', 'red'))
    # end if

    model = UNet(in_channels=3, n_classes=len(classes) + 1, depth=3, wf=6, padding=True, batch_norm=False, up_mode='upsample')

    # get the location of the final graph name
    graphLocs = glob.glob(os.path.join(GRAPHS_LOC, '*.pt'))
    graphLocs = natsort.natsorted(graphLocs)
    finalGraphLoc = graphLocs[-1]
    # load the final graph
    model.load_state_dict(torch.load(finalGraphLoc))

    model = model.to(device)
    model.eval()

    # setup params for thresh and morph opening
    backgroundThresh = 255 // 2
    openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    framesOfDetections2d: List[Utils.FrameOfDetections2D] = []

    ### compute detection boxes, scores, and classes on prediction images from test data ###

    print('\n' + 'beginning testing . . . ' + '\n')

    with torch.no_grad():
        for i, (frameImages, maskImages, tokenIds) in enumerate(tqdm(testDataLoader)):
            # move frame images to GPU if applicable
            frameImages = frameImages.to(device)  # frameImages shape = [ batchSize, 3, height, width ], dtype = torch.float32

            # get net output
            predImages = model(frameImages)  # predImages shape = [ batchSize, 2 (numClasses + 1 for bg), height, width ], dtype = torch.float32

            # run softmax on prediction image
            predImages = torch.nn.functional.softmax(predImages, dim=1)

            # move prediction image to CPU and convert from PyTorch tensor to numpy array
            predImages = predImages.cpu().numpy()
            # float32 in range 0.0 to 1.0 -> uint8 in range 1 to 255
            predImages = np.round(predImages * 255).astype(np.uint8)

            # only use the 1st channel, car (don't use the 0th channel, background)
            predImages = predImages[:, 1, :, :]

            # perform opening
            assert len(tokenIds) == len(predImages)
            for j, predImage in enumerate(predImages):
                # threshold and open image, save result as new image pred image processed
                _, predImageProc = cv2.threshold(predImage, backgroundThresh, 255, cv2.THRESH_BINARY)
                predImageProc = cv2.morphologyEx(predImageProc, cv2.MORPH_OPEN, openingKernel)

                # visualize validation, only every 50th iteration of the dataloader, and only for the 1st image within each predImages tensor
                if i % 50 == 0 and j == 0:
                    Utils.saveVisualizeImages(frameImages, maskImages, predImage, predImageProc, os.path.join(VIS_TEST_DIR, '7_' + str(i) + '.png'))
                # end if

                boxes, scores, classifications = Utils.calcDetectionBoxes2d(predImageProc, predImage, 'car')  # for now, class is always car

                framesOfDetections2d.append(Utils.FrameOfDetections2D(tokenIds[j], np.array(boxes), scores, classifications))
            # end for

        # end for
    # end with

    # computing prediction boxes ###############################

    print('\n' + 'computing prediction boxes . . . ' + '\n')

    predBox3Ds = []
    for frameOfDetections2d in tqdm(framesOfDetections2d):
        boxes = frameOfDetections2d.boxes2d
        scores = frameOfDetections2d.scores
        classifications = frameOfDetections2d.classifications

        # go from frame token ID to ego pose of lidar top data
        sample = level5data.get('sample', frameOfDetections2d.frameId)
        sampleTopLidarToken = sample['data']['LIDAR_TOP']
        lidarTopData = level5data.get('sample_data', sampleTopLidarToken)
        egoPose = level5data.get('ego_pose', lidarTopData['ego_pose_token'])

        # egoPose is a dictionary with 4 fields:
        # 'rotation': (4 floats), 'translation': (3 floats), 'token': (string), 'timestamp': (float)

        egoPoseRotMtx = pyquaternion.Quaternion(egoPose['rotation']).rotation_matrix

        carToGlobalTransMtx = Utils.makeTransformMatrix(egoPoseRotMtx, np.array(egoPose['translation']))

        voxelToCarTransMtx = Utils.makeCarToVoxelTransformMatrix(tdvImageShape, lidarRange, zOffset)

        invVoxelToCarTransMtx = np.linalg.inv(voxelToCarTransMtx)

        voxelToGlobalTransMtx = np.dot(carToGlobalTransMtx, invVoxelToCarTransMtx)

        # numBoxesInFrame, 4 points, 2 numbers per point -> numBoxesInFrame x 4, 2
        boxes = boxes.reshape(-1, 2)

        # numBoxesInFrame x 4, 2 -> 2, numBoxesInFrame x 4
        boxes = boxes.transpose()

        boxes = np.vstack((boxes, np.zeros(boxes.shape[1])))

        boxes = np.vstack((boxes, np.ones(boxes.shape[1])))
        boxes = np.dot(voxelToGlobalTransMtx, boxes)
        boxes = boxes[:3, :]

        # Fill in the z axis values (heights) of all boxes in 3d space.  Since we effectively lost the z-axis data
        # when we translated to a 2d top-down view, we have to suppose all box heights are the same as the vehicle height.
        boxes[2, :] = egoPose['translation'][2]  # egoPose['translation'][2] == vehicle height

        # 3, numBoxesInFrame * 4 -> numBoxesInFrame * 4, 3
        boxes = boxes.transpose()

        # numBoxesInFrame * 4, 3 -> numBoxesInFrame, 4 points per box, 3 numbers per point
        boxes = boxes.reshape((-1, 4, 3))

        boxCenters = boxes.mean(axis=1)

        # we don't have a way to know the height of our boxes, use approximate height of a typical car for every height since that is most common
        carHeight = 1.75

        # increment every z box center by 1/2 a car's height
        boxCenters[:, 2] += carHeight / 2

        widths = np.zeros(len(boxes))
        lengths = np.zeros(len(boxes))
        for i, box in enumerate(boxes):
            lengths[i] = np.linalg.norm(box[1] - box[0])
            widths[i] = np.linalg.norm(box[2] - box[1])
        # end for

        boxDims = np.zeros_like(boxCenters)
        boxDims[:, 0] = widths
        boxDims[:, 1] = lengths
        boxDims[:, 2] = carHeight

        # for each box in the current frame . . .
        for i in range(len(boxes)):
            translation = boxCenters[i]  # translation can be thought of as box center of mass

            size = boxDims[i]

            classification = classifications[i]

            # determine the rotation of the box
            rotVec = boxes[i][0] - boxes[i][1]

            # divide all elements in the rotation vector by the length of the rotation vector
            rotVec /= np.linalg.norm(rotVec)

            # make a rotation matrix from the rotation vector
            rotMtx = np.array([[rotVec[0], -rotVec[1], 0],
                               [rotVec[1],  rotVec[0], 0],
                               [   0     ,     0     , 1]])

            # make a quaternion from the rotation matrix
            quat = pyquaternion.Quaternion(matrix=rotMtx)

            detectionScore = float(scores[i])

            box3d = Box3D(sample_token=frameOfDetections2d.frameId,
                          translation=list(translation),
                          size=list(size),
                          rotation=list(quat),
                          name=classification,
                          score=detectionScore)

            predBox3Ds.append(box3d)
        # end for
    # end for

    print('\n' + 'making submission file . . . ' + '\n')

    # based on https://www.kaggle.com/meaninglesslives/lyft3d-inference-prediction-visualization
    subDict = collections.OrderedDict()

    with open(SAMPLE_SUB_FILE_LOC) as f:
        for i, line in enumerate(f):
            if i == 0: continue

            line = line.rstrip()  # remove trailing newline char
            line = line[:-1]   # remove last char (comma)

            subDict[line] = ''
        # end for
    # end with

    for predBox3D in predBox3Ds:
        # calculate yaw from the 1st element of rotation
        yaw = 2 * np.arccos(predBox3D.rotation[0])
        # build up the prediction string
        pred = str(predBox3D.score / 255.0) + ' ' + str(predBox3D.center_x) + ' ' + str(predBox3D.center_y) + ' ' + \
               str(predBox3D.center_z) + ' ' + str(predBox3D.width) + ' ' + str(predBox3D.length) + ' ' + \
               str(predBox3D.height) + ' ' + str(yaw) + ' ' + str(predBox3D.name)

        if subDict[predBox3D.sample_token] == '':
            subDict[predBox3D.sample_token] = pred
        else:
            subDict[predBox3D.sample_token] += ' ' + pred
        # end if
    # end for

    with open(os.path.join(CSV_LOC, KAGGLE_SUB_FILE_NAME), 'w') as subFile:
        subFile.write('Id,PredictionString' + '\n')
        for tokenId, predStr in subDict.items():
            subFile.write(str(tokenId) + ',' + str(predStr) + '\n')
        # end for
    # end with

    endTime = time.time()
    elapsedTime = endTime - startTime
    hrMinSecString = Utils.getHrMinSecString(elapsedTime)

    print('\n' + 'entire process took: ' + str(hrMinSecString))

    print('\n' + 'done !! ' + '\n')

    # end for

# end function

if __name__ == '__main__':
    main()



