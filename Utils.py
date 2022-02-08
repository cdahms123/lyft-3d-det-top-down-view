# Utils.py

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D

import torch
import torch.nn.functional
import torchvision

import numpy as np
import cv2
import pyquaternion
import pandas as pd
import pathlib
from typing import List, Tuple, Dict

class FrameOfDetections2D(object):
    def __init__(self, frameId: str, boxes2d: np.ndarray, scores: List[int], classifications: List[str]):
        self.frameId = frameId
        self.boxes2d = boxes2d
        self.scores = scores
        self.classifications = classifications
    # end function
# end function

def makeTdvImagesForFrame(level5data: LyftDataset, frameId: str, classes: List[str], tdvImageShape: Tuple[int, int, int],
                          lidarRange: Tuple[float, float, float], zOffset: float):
    # get lidar points from frame ID
    sample: dict = level5data.get('sample', frameId)
    lidarTopId: str = sample['data']['LIDAR_TOP']
    lidarFilePathObj: pathlib.Path = level5data.get_sample_data_path(lidarTopId)

    try:
        lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePathObj)
    except Exception as ex:
        if frameId == '9cb04b1a4d476fd0782431764c7b55e91c6dbcbc6197c3dab3e044f13d058011':
            pass
        else:
            print('\n' + 'error loading point cloud for frame ID ' + str(frameId) + ', ex: ' + str(ex))
        # end if
        return None, None
    # end try

    # transform points from sensor to car frame of reference
    lidarTopData: dict = level5data.get('sample_data', lidarTopId)
    calSensorData: dict = level5data.get('calibrated_sensor', lidarTopData['calibrated_sensor_token'])
    sensToCarRotMtx: np.ndarray = pyquaternion.Quaternion(calSensorData['rotation']).rotation_matrix
    sensToCarTranslationMtx: np.ndarray = np.array(calSensorData['translation'])
    sensToCarTransformMtx: np.ndarray = makeTransformMatrix(sensToCarRotMtx, sensToCarTranslationMtx)
    lidarPointCloud.transform(sensToCarTransformMtx)
    # intensity is always 100, so remove it
    points: np.ndarray = lidarPointCloud.points[:3, :]

    # points is a numpy arrayof type float32 in the following format
    #                   cols
    #         0   1   2   3   4   5   . . . n
    #      0  x1, x2, x3, x4, x5, x6  . . . xn
    # rows 1  y1, y2, y3, y4, y5, y6  . . . yn
    #      2  z1, z2, z3, z4, z5, z6  . . . zn

    frameImage = makeTdvFrameImage(points, tdvImageShape, lidarRange, zOffset)

    # make mask image

    # get ground truth boxes from lidar token
    gndTrBoxes: List[Box] = level5data.get_boxes(lidarTopId)

    # throw out all ground truth boxes except for cars
    gndTrBoxes = [box for box in gndTrBoxes if box.name == 'car']

    egoPoseData: dict = level5data.get('ego_pose', lidarTopData['ego_pose_token'])
    moveBoxesFromWorldSpaceToCarSpace(gndTrBoxes, egoPoseData)

    # note makeImage must be 3-channel even though it's grayscale due to some of the utility functions that drawBoxesOnMaskImage calls requiring a 3-channel image
    maskImage = np.zeros(tdvImageShape, np.uint8)

    drawBoxesOnTdvImage(maskImage, tdvImageShape, gndTrBoxes, (255, 255, 255), -1, lidarRange, zOffset)

    maskImage = cv2.cvtColor(maskImage, cv2.COLOR_BGR2GRAY)

    return frameImage, maskImage
# end function

def makeTransformMatrix(rotMtx: np.ndarray, translateMtx: np.ndarray) -> np.ndarray:
    """
    :param rotMtx: a 3x3 rotation matrix
    :param translateMtx: a 1-d 3-element numpy array representing translation [x, y, z]
    :return: a 4x4 numpy array, the transformation matrix, format is:
              rot  rot  rot  trans
              rot  rot  rot  trans
              rot  rot  rot  trans
              0.0  0.0  0.0   1.0 (last row unchanged from np.eye(4))
    """
    assert type(rotMtx) == np.ndarray
    assert len(rotMtx.shape) == 2     # rotMtx must be a 2d array
    assert len(rotMtx) == 3           # rotMtx must be 3 x 3
    assert len(rotMtx[0]) == 3

    assert type(translateMtx) == np.ndarray
    assert len(translateMtx.shape) == 1  # translationMtx must be a 1d array
    assert len(translateMtx) == 3        # translationMtx must be 3 elements long

    transformMatrix = np.eye(4)
    # copy rotMtx into top left 3x3 locations
    transformMatrix[:3, :3] = rotMtx
    # copy translateMtx vertically into right-most column
    transformMatrix[:3, 3] = translateMtx
    # note the last row is unchanged from np.eye

    return transformMatrix
# end function

def makeTdvFrameImage(points: np.ndarray, tdvImageShape: Tuple[int, int, int], lidarRange: Tuple[float, float, float], zOffset: float) -> np.ndarray:
    assert points.shape[0] == 3     # points must be 3 rows by n cols

    points = carPointsToVoxelPoints(points, tdvImageShape, lidarRange, zOffset)

    # change from rows being x, y, z to cols being x, y, z
    points = points.transpose()

    imageWidth = tdvImageShape[0]
    imageHeight = tdvImageShape[1]
    numChannels = tdvImageShape[2]

    image = np.zeros(tdvImageShape, np.uint8)
    for point in points:
        # break out x, y, and z from point
        x = point[0]
        y = point[1]
        z = point[2]

        if x < 0 or x >= imageWidth or y < 0 or y >= imageHeight or z < 0 or z >= numChannels: continue

        # note x and y are flipped (numpy uses height, width)
        value = image[y][x][z] + 16
        if value > 255: value = 255
        image[y][x][z] = value
    # end for

    return image
# end function

def carPointsToVoxelPoints(points: np.ndarray, tdvImageShape: Tuple[int, int, int], lidarRange: Tuple[float, float, float], zOffset: float) -> np.ndarray:
    assert points.shape[0] == 3  # points must be 3 rows by n cols

    carToVoxelTransMtx = makeCarToVoxelTransformMatrix(tdvImageShape, lidarRange, zOffset)

    # matrix multiply points by carToVoxelTransMtx to voxelize points
    # add a bottom row of dummy 1s
    points = np.vstack((points, np.ones(points.shape[1])))
    # matrix multiplication
    points = np.dot(carToVoxelTransMtx, points)
    # strip off 4th row of dummy 1s
    points = points[:3, :]

    # round to int64
    points = np.round(points).astype(np.int64)

    return points
# end function

def makeCarToVoxelTransformMatrix(tdvImageShape: Tuple[int, int, int], lidarRange: Tuple[float, float, float], zOffset: float) -> np.ndarray:
    imageWidth = tdvImageShape[0]
    imageHeight = tdvImageShape[1]
    numChannels = tdvImageShape[2]

    lidarRangeX = lidarRange[0]
    lidarRangeY = lidarRange[1]
    lidarRangeZ = lidarRange[2]

    transformMtx = np.eye(4, dtype=np.float32)

    # populate rotation portion
    transformMtx[0, 0] = imageWidth / lidarRangeX
    transformMtx[1, 1] = imageHeight / lidarRangeY
    transformMtx[2, 2] = numChannels / lidarRangeZ

    # populate translation portion
    transformMtx[0, 3] = imageWidth / 2
    transformMtx[1, 3] = imageHeight / 2
    transformMtx[2, 3] = (numChannels / 2) + (zOffset * (numChannels / lidarRangeZ))

    return transformMtx
# end function

# ToDo: these next 2 functions seem very similar, can these be combined ??

def moveBoxesFromWorldSpaceToCarSpace(boxes: List[Box], egoPoseData: Dict) -> None:
    # Note: changes input boxes

    translationMtx: np.ndarray = np.array(egoPoseData['translation'])
    translationMtx = -translationMtx

    rotationQuat = pyquaternion.Quaternion(egoPoseData['rotation'])
    rotationQuat = rotationQuat.inverse

    for box in boxes:
        box.translate(translationMtx)
        box.rotate(rotationQuat)
    # end for
# end function

def moveBoxFromWorldSpaceToSensorSpace(level5data: LyftDataset, box: Box, lidarTopData: dict) -> Box:

    box = box.copy()

    # world space to car space
    egoPoseData: dict = level5data.get('ego_pose', lidarTopData['ego_pose_token'])
    box.translate(-np.array(egoPoseData['translation']))
    box.rotate(pyquaternion.Quaternion(egoPoseData['rotation']).inverse)

    # car space to sensor space
    calSensorData: dict = level5data.get('calibrated_sensor', lidarTopData['calibrated_sensor_token'])
    box.translate(-np.array(calSensorData['translation']))
    box.rotate(pyquaternion.Quaternion(calSensorData['rotation']).inverse)

    return box
# end function

def drawBoxesOnTdvImage(tdvImage: np.ndarray, tdvImageShape: Tuple[int, int, int], boxes: List[Box], openCvColor: Tuple[int, int, int],
                        contourThickness: int, lidarRange: Tuple[float, float, float], zOffset: float=0.0) -> None:
    # Note: pass in contourThickness of -1 to fill in contours if desired

    assert tdvImage.shape[2] == 3    # verify 3-channel

    for box in boxes:
        # TDV, so only need bottom corners
        cornerPts: np.ndarray = box.bottom_corners()

        # cornerPts is in the following format:
        #              cols
        #          0   1   2   3
        #      0  x0  x1  x2  x3
        # rows 1  y0  y1  y2  y3
        #      2  z0  z1  z2  z3

        # change to voxel coordinates
        voxCornerPts = carPointsToVoxelPoints(cornerPts, tdvImageShape, lidarRange, zOffset)
        # keep xs and yz, drop zs
        voxCornerPts = voxCornerPts[:2, :]
        # transpose so can pass into drawContours below
        voxCornerPts = voxCornerPts.transpose()

        cv2.drawContours(tdvImage, [voxCornerPts], 0, openCvColor, contourThickness)
    # end for
# end function

def saveVisualizeImages(frameImages: torch.Tensor, maskImages: torch.Tensor, predImage: np.ndarray, predImageProc: np.ndarray, saveImageLoc: str) -> None:
    assert frameImages.dtype == torch.float32 and len(frameImages.shape) == 4
    assert maskImages.dtype == torch.int64 and len(maskImages.shape) == 3
    assert predImage.dtype == np.uint8 and len(predImage.shape) == 2
    assert predImageProc.dtype == np.uint8 and len(predImageProc.shape) == 2

    # convert frame images PyTorch tensor to a 3-channel BGR uint8 OpenCV image (using only the 1st image from the images PyTorch tensor)

    # get the first frame image in the batch
    frameImage = frameImages[0]

    # convert from PyTorch tensor to numpy array
    frameImage = torchvision.transforms.ToPILImage()(frameImage)

    # convert from float32 to uint8
    frameImage = np.array(frameImage).astype(np.uint8)

    # RGB -> BGR
    frameImage = cv2.cvtColor(frameImage, cv2.COLOR_RGB2BGR)

    # convert mask images PyTorch tensor to a 3-channel BGR uint8 OpenCV image (using only the 1st image from the images PyTorch tensor)

    # get the first mask image in the batch
    maskImage = maskImages[0]

    # convert from int64 to uint8
    maskImage = maskImage.type(torch.uint8)

    # convert from PyTorch tensor to numpy array
    maskImage = torchvision.transforms.ToPILImage()(maskImage)
    maskImage = np.array(maskImage)

    # turn 1s into 255s so they are visible (leave 0s as 0s)
    maskImage = np.where(maskImage==1, 255, maskImage)

    # save images together

    # make all images the same shape (3-channel) and horizontally stack for easier display

    # verify frame image is already 3-channel
    assert len(frameImage.shape) == 3 and frameImage.shape[2] == 3
    # convert other 3 images to 3-channel
    maskImage = cv2.cvtColor(maskImage, cv2.COLOR_GRAY2BGR)
    predImage = cv2.cvtColor(predImage, cv2.COLOR_GRAY2BGR)
    predImageProc = cv2.cvtColor(predImageProc, cv2.COLOR_GRAY2BGR)

    # make a 1 pixel wide image to put in-between the others to make the image borders more clear
    vert1PxImage = np.full((frameImage.shape[0], 1, 3), (255, 0, 0), np.uint8)

    allImagesHStack = np.hstack((frameImage, vert1PxImage, predImage, vert1PxImage, predImageProc, vert1PxImage, maskImage))

    cv2.imwrite(saveImageLoc, allImagesHStack)
# end function

def getGndTrBox3Ds(level5data: LyftDataset, frameId: str) -> List[Box3D]:
    gndTrBox3Ds: List[Box3D] = []

    sample = level5data.get('sample', frameId)
    annIds = sample['anns']

    for annId in annIds:
        annData: dict = level5data.get('sample_annotation', annId)
        box3d = Box3D(sample_token=frameId,
                      translation=annData['translation'],
                      size=annData['size'],
                      rotation=annData['rotation'],
                      name=annData['category_name'])  # classification name
        gndTrBox3Ds.append(box3d)
    # end for

    return gndTrBox3Ds
# end function

def calcDetectionBoxes2d(predImageProc: np.ndarray, predImage: np.ndarray, className: str) -> Tuple[List[np.ndarray], List[int], List[str]]:
    assert predImageProc.shape == (500, 500)
    assert predImageProc.dtype == np.uint8
    assert predImage.shape == (500, 500)
    assert predImage.dtype == np.uint8

    boxes: List[np.ndarray] = []
    scores = []
    classifications = []

    contours, _ = cv2.findContours(predImageProc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        # get the min area rectangle surrounding the contour, note this is a rotated rect
        rotRect = cv2.minAreaRect(contour)
        # get the 4 points of the min area rotated rect
        rotRectPts: np.ndarray = cv2.boxPoints(rotRect)

        # adjust rotated rect points so 0 to 1 is always length and 1 to 2 is always width
        zeroToOne = np.linalg.norm(rotRectPts[1] - rotRectPts[0])
        oneToTwo = np.linalg.norm(rotRectPts[2] - rotRectPts[1])
        if zeroToOne > oneToTwo:
            pass
        else:
            rotRectPts = np.roll(rotRectPts, 1, axis=0)
        # end if

        # get the center x and y of the rotated rect
        rectCenterPt = rotRect[0]
        centerY = round(rectCenterPt[1])
        centerX = round(rectCenterPt[0])

        # get the pixel value from the unprocessed (before thresholding/opening) image at the center of the contour
        centerValue = predImage[centerY, centerX]

        # pass on candidates with very low probability
        if centerValue < 0.01:
            continue
        # end if

        boxes.append(rotRectPts)
        scores.append(int(centerValue))
        classifications.append(className)
    # end for

    return boxes, scores, classifications
# end function

def getPredBoxes(df: pd.DataFrame, idx: int) -> List[Box]:

    frameId = df.iloc[idx]['Id']

    predString = df.iloc[idx]['PredictionString']

    if not isinstance(predString, str) or predString == '':
        return []
    # end if

    predTokens = predString.split(' ')

    assert len(predTokens) % 9 == 0

    predBoxes = []
    for i in range(0, len(predTokens), 9):
        confi = float(predTokens[i + 0])
        x = float(predTokens[i + 1])
        y = float(predTokens[i + 2])
        z = float(predTokens[i + 3])
        width = float(predTokens[i + 4])
        length = float(predTokens[i + 5])
        height = float(predTokens[i + 6])
        yaw = float(predTokens[i + 7])
        classification = str(predTokens[i + 8])

        box = Box(center=[x, y, z],
                  size=[width, length, height],
                  orientation=pyquaternion.Quaternion(axis=[0, 0, 1], radians=yaw),
                  score=confi,
                  name=classification,
                  token=frameId)
        predBoxes.append(box)
    # end for

    return predBoxes
# end function

def addLineToPlotlyLines(point1, point2, xLines: List, yLines: List, zLines: List) -> None:
    xLines.append(point1[0])
    xLines.append(point2[0])
    xLines.append(None)

    yLines.append(point1[1])
    yLines.append(point2[1])
    yLines.append(None)

    zLines.append(point1[2])
    zLines.append(point2[2])
    zLines.append(None)
# end function

def getHrMinSecString(seconds: float) -> str:
    hrs = seconds // 3600
    mins = seconds % 3600 // 60
    secs = seconds % 3600 % 60
    hrMinSecString = str(hrs) + ' hours, ' + str(mins) + ' mins, ' + '{:.2f}'.format(secs) + ' secs'
    return hrMinSecString
# end function



