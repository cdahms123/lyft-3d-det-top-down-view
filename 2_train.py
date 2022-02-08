# 2_train.py

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
import time
from tqdm import tqdm
from typing import List
from termcolor import colored
import pprint

VIS_TRAIN_DIR = os.path.join(os.getcwd(), 'vis_train')
LYFT_TRAIN_DATASET_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'train')

# TDV = Top Down View
TDV_TRAIN_IMAGES_LOC = os.path.join(os.getcwd(), 'tdv_train_images')
TDV_VAL_IMAGES_LOC = os.path.join(os.getcwd(), 'tdv_val_images')

GRAPHS_LOC = 'graphs'
GRAPH_NAME_BEG = 'unet_epoch_'
GRAPH_NAME_END = '.pt'

CSV_LOC = os.path.join(os.getcwd(), 'csv')

NUM_EPOCHS = 5     # use 25 epochs for best results, 5 for quick test results
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE = 8

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    # time bookkeeping
    startTime = time.time()

    # make the directory for saving training visualization images to, if it does not exist already
    os.makedirs(VIS_TRAIN_DIR, exist_ok=True)

    # load training data
    print('\n' + 'loading training data . . . ' + '\n')
    level5data = LyftDataset(data_path=LYFT_TRAIN_DATASET_LOC, json_path=os.path.join(LYFT_TRAIN_DATASET_LOC, 'data'), verbose=True)

    classes = ['car']

    # level5data.scene is a list (180 elements long) of dictionaries
    print('\n' + 'type(level5data.scene) = ' + str(type(level5data.scene)))  # class 'list'
    print('len(level5data.scene) = ' + str(len(level5data.scene)))  # 180 elements long

    # a single 'scene' is a 25-45 second snippet of a car's journey

    # keys for each scene dictionary are description, first_sample_token, last_sample_token, log_token, name, nbr_samples, token
    print('\n' + 'type(level5data.scene[0]) = ' + str(type(level5data.scene[0])))
    print('level5data.scene[0]: ')
    pprint.pprint(level5data.scene[0])
    print('')

    # Use hosts a007, a008, and a009 for validation, all others for training.
    # This is a 40 / 180 (0.22) validation split, i.e. an approximately 78% train / 22% val split
    valHostNames = ['host-a007', 'host-a008', 'host-a009']

    trainFrameIds = []
    valFrameIds = []
    # for each scene . . .
    for scene in level5data.scene:
        # need to get the host name (ex. 'host-a101') for the current scene, we can get this from the name
        # name will be for example 'host-a101-lidar0-1241893239199111666-1241893264098084346'
        sceneName = scene['name']
        # but we need just the 'host-a101' part, so,
        # make a list of the first 2 tokens split by '-', ex. will be ['host', 'a101']
        twoTokenList = sceneName.split('-')[:2]
        # join, separated by a '-', now will be ex. 'host-a101'
        hostName = '-'.join(twoTokenList)

        # if the host name is in the validation host names list (defined above) . . .
        if hostName in valHostNames:
            frameId = scene['first_sample_token']
            # loop through all the frame IDs in the current scene and add them to the validation frame IDs list
            while frameId is not None and frameId != '':
                valFrameIds.append(frameId)
                # advance frame ID to the next frame (will be None or an empty string if there are no more frames in this trip)
                sample = level5data.get('sample', frameId)
                frameId = sample['next']
            # end while
        else:  # else if the host name is not in the validation host names list then it's a training scene
            frameId = scene['first_sample_token']
            # loop through all the frame IDs in the current scene and add them to the training frame IDs list
            while frameId is not None and frameId != '':
                trainFrameIds.append(frameId)
                # advance frame ID to the next frame (will be None or an empty string if there are no more frames in this trip)
                sample = level5data.get('sample', frameId)
                frameId = sample['next']
            # end while
        # end if
    # end for

    tdvImageShape = (500, 500, 3)
    lidarRange = (200.0, 200.0, 4.5)
    zOffset = -2.0

    ### write train and validation TDV images #######################

    if os.path.isdir(TDV_TRAIN_IMAGES_LOC) and os.path.isdir(TDV_VAL_IMAGES_LOC):
        print('\n' + 'TDV directories already exist, skipping over making TDV images' + '\n')
    else:
        # if either the TDV train image directory or the TDV validation image directory does not exist, create it
        os.makedirs(TDV_TRAIN_IMAGES_LOC, exist_ok=True)
        os.makedirs(TDV_VAL_IMAGES_LOC, exist_ok=True)

        print('\n' + 'making training TDV images . . . ' + '\n')
        for frameId in tqdm(trainFrameIds):
            frameImage, maskImage = Utils.makeTdvImagesForFrame(level5data, frameId, classes, tdvImageShape, lidarRange, zOffset)
            if frameImage is not None and maskImage is not None:
                cv2.imwrite(os.path.join(TDV_TRAIN_IMAGES_LOC, str(frameId) + '_frame.png'), frameImage)
                cv2.imwrite(os.path.join(TDV_TRAIN_IMAGES_LOC, str(frameId) + '_mask.png'), maskImage)
            # end if
        # end for

        print('\n' + 'making validation TDV images . . . ' + '\n')
        for frameId in tqdm(valFrameIds):
            frameImage, maskImage = Utils.makeTdvImagesForFrame(level5data, frameId, classes, tdvImageShape, lidarRange, zOffset)
            if frameImage is not None and maskImage is not None:
                cv2.imwrite(os.path.join(TDV_VAL_IMAGES_LOC, str(frameId) + '_frame.png'), frameImage)
                cv2.imwrite(os.path.join(TDV_VAL_IMAGES_LOC, str(frameId) + '_mask.png'), maskImage)
            # end if
        # end for
    # end if

    ### setup for training and validation ###########################

    # get file locs of training frame and mask images
    trainFrameFileLocs = sorted(glob.glob(os.path.join(TDV_TRAIN_IMAGES_LOC, '*_frame.png')))
    print('len(trainFrameFileLocs) = ' + str(len(trainFrameFileLocs)))
    trainMaskFileLocs = sorted(glob.glob(os.path.join(TDV_TRAIN_IMAGES_LOC, '*_mask.png')))
    print('len(trainMaskFileLocs) = ' + str(len(trainMaskFileLocs)))

    trainDataset = TdvImageDataset(trainFrameFileLocs, trainMaskFileLocs)
    print('len(trainDataset) = ' + str(len(trainDataset)))

    trainDataLoader = torch.utils.data.DataLoader(trainDataset, TRAIN_BATCH_SIZE, shuffle=True)
    print('len(trainDataLoader) = ' + str(len(trainDataLoader)))

    # get file locs of validation frame and mask images
    valFrameFileLocs = sorted(glob.glob(os.path.join(TDV_VAL_IMAGES_LOC, '*_frame.png')))
    print('len(valFrameFileLocs) = ' + str(len(valFrameFileLocs)))
    valMaskFileLocs = sorted(glob.glob(os.path.join(TDV_VAL_IMAGES_LOC, '*_mask.png')))
    print('len(valMaskFileLocs) = ' + str(len(valMaskFileLocs)))

    valDataset = TdvImageDataset(valFrameFileLocs, valMaskFileLocs)
    print('\n' + 'len(valDataset) = ' + str(len(valDataset)))

    valDataLoader = torch.utils.data.DataLoader(valDataset, VAL_BATCH_SIZE, shuffle=False)
    print('\n' + 'len(valDataLoader) = ' + str(len(valDataLoader)))

    # get device (cuda or cpu)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(colored('using GPU', 'green'))
    else:
        device = torch.device('cpu')
        print(colored('CUDA does not seem to be available, using CPU', 'red'))
    # end if

    # ToDo: may need to change these UNet parameters more for best results ??
                              # num classes + 1 for background
    # model = UNet(in_channels=3, n_classes=len(classes)+1, depth=4, wf=5, padding=True, up_mode='upsample')
    model = UNet(in_channels=3, n_classes=len(classes) + 1, depth=3, wf=6, padding=True, batch_norm=False, up_mode='upsample')
    model = model.to(device)

    finalGraphName = GRAPH_NAME_BEG + str(NUM_EPOCHS) + GRAPH_NAME_END
    finalGraphLoc = os.path.join(GRAPHS_LOC, finalGraphName)

    ### training / validation #######################################

    if os.path.isfile(finalGraphLoc):
        print('\n' + 'final graph already exists, loading trained graph from file instead of training' + '\n')
        model.load_state_dict(torch.load(finalGraphLoc))
    else:
        print('\n' + 'beginning training . . . ' + '\n')

        os.makedirs(GRAPHS_LOC, exist_ok=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainLosses = []

        for epoch in range(1, NUM_EPOCHS + 1):

            print('epoch ' + str(epoch))
            epochLosses = []

            # training

            model.train()
            for i, (frameImages, maskImages, frameIds) in enumerate(tqdm(trainDataLoader)):
                # move frame and mask images to GPU if applicable
                frameImages = frameImages.to(device)    # frameImages shape = [ batchSize, 3, height, width ], dtype = torch.float32
                # print(frameImages.shape)
                maskImages = maskImages.to(device)      # maskImages shape = [ batchSize, height, width ], dtype = torch.int64
                # print(maskImages.shape)

                # clear gradients from the previous step
                optimizer.zero_grad()

                # get net output
                predImages = model(frameImages)     # predImages shape = [ batchSize, 2 (numClasses + 1 for bg), height, width ], dtype = torch.float32
                # print(predImages.shape)

                # calculate loss
                loss = torch.nn.functional.cross_entropy(predImages, maskImages)
                # call backward to compute gradients
                loss.backward()
                # update parameters using gradients
                optimizer.step()

                # append the current classification loss to the list of epoch losses
                epochLosses.append(loss.detach().cpu().numpy())

                # save visualize image every x iterations
                if i % 1000 == 0:
                    # Note: everything inside this if block is only for visualization, code has to be here to be
                    #       similar to validation(below) so both can use the same visualization function in Utils

                    # run softmax on prediction images (will change values to 0.0 to 1.0 range)
                    predImages = torch.nn.functional.softmax(predImages, dim=1)

                    # move prediction images to CPU and convert from PyTorch tensor to numpy array
                    predImages = predImages.detach().cpu().numpy()
                    # float32 in range 0.0 to 1.0 -> uint8 in range 1 to 255
                    predImages = np.round(predImages * 255).astype(np.uint8)

                    # only use the 1st channel, car (don't use the 0th channel, background)
                    predImages = predImages[:, 1, :, :]

                    predImage = predImages[0]

                    # threshold and open image, save result as new image, pred image processed
                    backgroundThresh = 255 // 2
                    _, predImageProc = cv2.threshold(predImage, backgroundThresh, 255, cv2.THRESH_BINARY)
                    openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    predImageProc = cv2.morphologyEx(predImageProc, cv2.MORPH_OPEN, openingKernel)
                    Utils.saveVisualizeImages(frameImages, maskImages, predImage, predImageProc, os.path.join(VIS_TRAIN_DIR, '5_' + str(epoch) + '_' + str(i) + '_' + '.png'))
                # end if
            # end for

            print('loss: ' + str(np.mean(epochLosses)))

            trainLosses.extend(epochLosses)

            graphFileName = GRAPH_NAME_BEG + str(epoch) + GRAPH_NAME_END
            graphFileLoc = os.path.join(GRAPHS_LOC, graphFileName)
            torch.save(model.state_dict(), graphFileLoc)

            # validation

            model.eval()
            valLosses = []
            with torch.no_grad():
                for i, (frameImages, maskImages, frameIds) in enumerate(tqdm(valDataLoader)):
                    # move frame images and mask images to GPU if applicable
                    frameImages = frameImages.to(device)  # frameImages shape = [ batchSize, 3, height, width ], dtype = torch.float32
                    maskImages = maskImages.to(device)  # maskImages shape = [ batchSize, height, width ], dtype = torch.int64

                    # get net output
                    predImages = model(frameImages)  # predImages shape = [ batchSize, 2 (numClasses + 1 for bg), height, width ], dtype = torch.float32

                    # compute loss and append to validation losses
                    loss = torch.nn.functional.cross_entropy(predImages, maskImages)
                    valLosses.append(loss.detach().cpu().numpy())
                # end for
            # end with
            print('\n' + 'mean of validation losses: ' + str(np.mean(valLosses)))

        # end for epoch in epochs

    # end big training/validation if

    # produce framesOfDetections2d list #############################

    # setup params for thresh and morph opening
    backgroundThresh = 255 // 2
    openingKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    framesOfDetections2d: List[Utils.FrameOfDetections2D] = []

    print('\n' + 'making framesOfDetections2d list . . . ' + '\n')

    with torch.no_grad():
        for i, (frameImages, maskImages, frameIds) in enumerate(tqdm(valDataLoader)):
            # move frame images and mask images to GPU if applicable
            frameImages = frameImages.to(device)   # frameImages shape = [ batchSize, 3, height, width ], dtype = torch.float32
            maskImages = maskImages.to(device)     # maskImages shape = [ batchSize, height, width ], dtype = torch.int64

            # get net output
            predImages = model(frameImages)     # predImages shape = [ batchSize, 2 (numClasses + 1 for bg), height, width ], dtype = torch.float32

            # run softmax on prediction images (will change values to 0.0 to 1.0 range)
            predImages = torch.nn.functional.softmax(predImages, dim=1)

            # move prediction images to CPU and convert from PyTorch tensor to numpy array
            predImages = predImages.cpu().numpy()
            # float32 in range 0.0 to 1.0 -> uint8 in range 1 to 255
            predImages = np.round(predImages * 255).astype(np.uint8)

            # only use the 1st channel, car (don't use the 0th channel, background)
            predImages = predImages[:, 1, :, :]

            # perform opening
            assert len(frameIds) == len(predImages)
            for j, predImage in enumerate(predImages):
                # threshold and open image, save result as new image pred image processed
                _, predImageProc = cv2.threshold(predImage, backgroundThresh, 255, cv2.THRESH_BINARY)
                predImageProc = cv2.morphologyEx(predImageProc, cv2.MORPH_OPEN, openingKernel)

                # visualize validation, only every 20th iteration of the dataloader, and only for the 1st image within each predImages tensor
                if i % 20 == 0 and j == 0:
                    Utils.saveVisualizeImages(frameImages, maskImages, predImage, predImageProc, os.path.join(VIS_TRAIN_DIR, '7_' + str(i) + '.png'))
                # end if

                boxes2d, scores, classifications = Utils.calcDetectionBoxes2d(predImageProc, predImage, 'car')   # for now, class is always car

                # ToDo: write function here to visualize frame and boxes, inputs should be frame ID and contour boxes

                framesOfDetections2d.append(Utils.FrameOfDetections2D(frameIds[j], np.array(boxes2d), scores, classifications))
            # end for
        # end for
    # end with

    ### compute prediction boxes ####################################

    print('\n' + 'computing prediction boxes . . . ' + '\n')

    # ToDo: figure out why pred boxes are rotated in many cases, suspect concern is in this next for loop !!

    valPredBox3Ds: List[Box3D] = []
    for frameOfDetections2d in tqdm(framesOfDetections2d):
        boxes = frameOfDetections2d.boxes2d
        scores = frameOfDetections2d.scores
        classifications = frameOfDetections2d.classifications

        # start last loop logging if applicable
        if not lastLoopLog.alreadyRan and len(boxes) == 3:
            lastLoopLog.logging = True
            lastLoopLog.logFile = open('last_loop_log.txt', 'w')
        # end if

        # go from frame token ID to ego pose of lidar top data
        sampleData: dict = level5data.get('sample', frameOfDetections2d.frameId)
        lidarTopId: str = sampleData['data']['LIDAR_TOP']
        lidarTopData: dict = level5data.get('sample_data', lidarTopId)
        egoPoseData: dict = level5data.get('ego_pose', lidarTopData['ego_pose_token'])
        lastLoopLog('egoPoseData: ', egoPoseData)
        lastLoopLog('egoPoseData[\'rotation\']: ', egoPoseData['rotation'])

        # egoPoseData is a dictionary with 4 fields:
        # 'rotation': (4 floats), 'translation': (3 floats), 'token': (string), 'timestamp': (float)

        egoPoseRotMtx = pyquaternion.Quaternion(egoPoseData['rotation']).rotation_matrix
        lastLoopLog('egoPoseRotMtx', egoPoseRotMtx)
        lastLoopLog('egoPoseData[\'translation\']', egoPoseData['translation'])

        carToGlobalTransMtx = Utils.makeTransformMatrix(egoPoseRotMtx, np.array(egoPoseData['translation']))
        lastLoopLog('carToGlobalTransMtx', carToGlobalTransMtx)

        voxelToCarTransMtx = Utils.makeCarToVoxelTransformMatrix(tdvImageShape, lidarRange, zOffset)
        lastLoopLog('voxelToCarTransMtx', voxelToCarTransMtx)

        invVoxelToCarTransMtx = np.linalg.inv(voxelToCarTransMtx)
        lastLoopLog('invVoxelToCarTransMtx', invVoxelToCarTransMtx)

        voxelToGlobalTransMtx = np.dot(carToGlobalTransMtx, invVoxelToCarTransMtx)
        lastLoopLog('voxelToGlobalTransMtx', voxelToGlobalTransMtx)

        lastLoopLog('boxes1', boxes)

        # numBoxesInFrame, 4 points, 2 numbers per point -> numBoxesInFrame x 4, 2
        boxes = boxes.reshape(-1, 2)
        lastLoopLog('boxes2', boxes)

        # numBoxesInFrame x 4, 2 -> 2, numBoxesInFrame x 4
        boxes = boxes.transpose()
        lastLoopLog('boxes3', boxes)

        # add z values for every box, make 0 for now, we will change in next few steps
        boxes = np.vstack((boxes, np.zeros(boxes.shape[1])))
        lastLoopLog('boxes4', boxes)

        # apply voxel to global transform to boxes
        # add row of dummy 1s so dot product does not blow up
        boxes = np.vstack((boxes, np.ones(boxes.shape[1])))
        # compute dot product
        boxes = np.dot(voxelToGlobalTransMtx, boxes)
        # remove row of dummy 1s
        boxes = boxes[:3, :]
        lastLoopLog('boxes5', boxes)

        # Fill in the z axis values (heights) of all boxes in 3d space.  Since we effectively lost the z-axis data
        # when we translated to a 2d top-down view, we have to suppose all box heights are the same as the vehicle height.
        boxes[2, :] = egoPoseData['translation'][2]  # egoPoseData['translation'][2] == vehicle height
        lastLoopLog('boxes6', boxes)

        # 3, numBoxesInFrame * 4 -> numBoxesInFrame * 4, 3
        boxes = boxes.transpose()
        lastLoopLog('boxes7', boxes)

        # numBoxesInFrame * 4, 3 -> numBoxesInFrame, 4 points per box, 3 numbers per point
        boxes = boxes.reshape((-1, 4, 3))
        lastLoopLog('boxes8', boxes)

        boxCenters = boxes.mean(axis=1)
        lastLoopLog('boxCenters', boxCenters)

        # we don't have a way to know the height of our boxes, use approximate height of a typical car for every height since that is most common
        carHeight = 1.75

        # increment every z box center by 1/2 a car's height
        boxCenters[:, 2] += carHeight / 2
        lastLoopLog('boxCenters', boxCenters)

        widths = np.zeros(len(boxes))
        lengths = np.zeros(len(boxes))
        for i, box in enumerate(boxes):
            lengths[i] = np.linalg.norm(box[1] - box[0])
            widths[i] = np.linalg.norm(box[2] - box[1])
        # end for
        lastLoopLog('widths', widths)
        lastLoopLog('lengths', lengths)

        boxDims = np.zeros_like(boxCenters)
        boxDims[:, 0] = widths
        boxDims[:, 1] = lengths
        boxDims[:, 2] = carHeight
        lastLoopLog('boxDims', boxDims)

        # for each box in the current frame . . .
        for i in range(len(boxes)):
            translation = boxCenters[i]   # translation can be thought of as box center of mass
            lastLoopLog('center', lengths)

            size = boxDims[i]
            lastLoopLog('size', size)

            classification = classifications[i]
            lastLoopLog('classification', classification)

            # ToDo: suspect rotation problem is in this next section, the quat passed in to Box3D rotation must be off

            rotVec = boxes[i][1] - boxes[i][0]
            lastLoopLog('rotVec1', rotVec)

            # divide all elements in the rotation vector by the length of the rotation vector
            # ToDo: need to understand this step better, i.e. explain what would go wrong if this step was skipped ??
            rotVec /= np.linalg.norm(rotVec)
            lastLoopLog('rotVev2', rotVec)

            # make a rotation matrix from the rotation vector
            # ToDo: need to understand how this is derrived, see https://en.wikipedia.org/wiki/Rotation_matrix
            rotMtx = np.array([[rotVec[0], -rotVec[1], 0],
                               [rotVec[1],  rotVec[0], 0],
                               [    0    ,     0     , 1]])
            lastLoopLog('rotMtx', rotMtx)

            # make a quaternion from the rotation matrix
            quat = pyquaternion.Quaternion(matrix=rotMtx)
            lastLoopLog('quat', quat)

            detectionScore = float(scores[i])
            lastLoopLog('detectionScore', detectionScore)

            box3d = Box3D(sample_token=frameOfDetections2d.frameId,
                          translation=list(translation),
                          size=list(size),
                          rotation=list(quat),
                          name=classification,
                          score=detectionScore)
            lastLoopLog('box3d', box3d)

            # stop last loop logging if applicable
            if lastLoopLog.logging:
                lastLoopLog.logging = False
                lastLoopLog.alreadyRan = True
                lastLoopLog.logFile.close()
            # end if

            valPredBox3Ds.append(box3d)
        # end for

    # end for

    # make a list of all the validation ground truth boxes (type Box3D)
    valGndTrBox3Ds = []
    for valFrameId in valFrameIds:
        currentFrameGndTrBox3Ds = Utils.getGndTrBox3Ds(level5data, valFrameId)
        valGndTrBox3Ds.extend(currentFrameGndTrBox3Ds)
    # end for
    print('\n' + 'len(valGndTrBox3Ds) = ' + str(len(valGndTrBox3Ds)))

    ### verify prediction IDs are the same as ground truth IDs

    # make a set of the prediction IDs
    valPredFrameIds = set()
    for box3d in valPredBox3Ds:
        valPredFrameIds.add(box3d.sample_token)
    # end for
    print('\n' + 'len(valPredFrameIds) = ' + str(len(valPredFrameIds)))

    # make a set of the ground truth frame IDs
    valGndTrFrameIds = set()
    for box3d in valGndTrBox3Ds:
        valGndTrFrameIds.add(box3d.sample_token)
    # end for
    print('len(valGndTrFrameIds) = ' + str(len(valGndTrFrameIds)))

    # verify that the validation prediction frame IDs are exactly the same as the validation ground truth frame IDs
    assert valPredFrameIds == valGndTrFrameIds

    print('\n' + 'validation prediction frame IDs verified to be the same as the validation ground truth frame IDs')

    os.makedirs(CSV_LOC, exist_ok=True)

    print('\n' + 'calculating mAP . . .')

    predsBigList = []
    for predBox3D in valPredBox3Ds:
        temp = dict()
        temp['sample_token'] = predBox3D.sample_token
        temp['translation'] = predBox3D.translation
        temp['size'] = predBox3D.size
        temp['rotation'] = predBox3D.rotation
        temp['name'] = predBox3D.name
        temp['score'] = predBox3D.score / 255.0
        predsBigList.append(temp)
    # end for

    gndTrBigList = []
    for gtBox3D in valGndTrBox3Ds:
        temp = dict()
        temp['sample_token'] = gtBox3D.sample_token
        temp['translation'] = gtBox3D.translation
        temp['size'] = gtBox3D.size
        temp['rotation'] = gtBox3D.rotation
        temp['name'] = gtBox3D.name
        gndTrBigList.append(temp)
    # end for

    # noinspection PyTypeChecker
    classNamesFromGtBigList = lyft_dataset_sdk.eval.detection.mAP_evaluation.get_class_names(gndTrBigList)
    print('\n' + 'Class_names = ', classNamesFromGtBigList)

    average_precisions = lyft_dataset_sdk.eval.detection.mAP_evaluation.get_average_precisions(gndTrBigList, predsBigList, classNamesFromGtBigList, 0.5)

    mAP = np.mean(average_precisions)
    print('\n' + 'Average per class mean average precision = ' + str(mAP))

    for class_id in sorted(list(zip(classNamesFromGtBigList, average_precisions.flatten().tolist()))):
        print(class_id)
    # end for

    valPredBoxesDict = dict()
    for predBox3d in valPredBox3Ds:
        # calculate yaw from the 1st element of rotation
        yaw = 2 * np.arccos(predBox3d.rotation[0])
        # build up the prediction string
        predString = str(predBox3d.score / 255.0) + ' ' + str(predBox3d.center_x) + ' ' + str(predBox3d.center_y) + ' ' + \
                      str(predBox3d.center_z) + ' ' + str(predBox3d.width) + ' ' + str(predBox3d.length) + ' ' + \
                      str(predBox3d.height) + ' ' + str(yaw) + ' ' + str(predBox3d.name)
        # if the current frame ID is not already in the Box3Ds dict, add the frame ID as the key
        if predBox3d.sample_token not in valPredBoxesDict:
            valPredBoxesDict[predBox3d.sample_token] = predString
        else:
            valPredBoxesDict[predBox3d.sample_token] += ' ' + predString
        # end if
    # end for

    print('\n' + 'writing results to ' + str(CSV_LOC) + ' . . . ' + '\n')

    with open(os.path.join(CSV_LOC, 'my_val.csv'), 'w') as trainCsvFile:
        trainCsvFile.write('Id,PredictionString' + '\n')
        for frameId, predStr in sorted(valPredBoxesDict.items()):
            trainCsvFile.write(str(frameId) + ',' + str(predStr) + '\n')
        # end for
    # end with

    ### complete timing bookkeeping #################################

    endTime = time.time()
    elapsedTime = endTime - startTime
    hrMinSecString = Utils.getHrMinSecString(elapsedTime)
    print('\n' + 'entire process took: ' + hrMinSecString)
    print('\n' + 'done !!' + '\n')
# end function

def lastLoopLog(varName: str, var):
    if lastLoopLog.alreadyRan: return
    if not lastLoopLog.logging: return

    lastLoopLog.logFile.write('\n')
    if isinstance(var, np.ndarray):
        lastLoopLog.logFile.write(varName + '.shape: ')
        lastLoopLog.logFile.write(str(var.shape))
        lastLoopLog.logFile.write('\n')
    # end if
    lastLoopLog.logFile.write(varName + ': ' + '\n')
    lastLoopLog.logFile.write(str(var))
    lastLoopLog.logFile.write('\n')
# end function

# set initial state of lastLoopLog function member variables
lastLoopLog.logging = False
lastLoopLog.logFile = None
lastLoopLog.alreadyRan = False

if __name__ == '__main__':
    main()



