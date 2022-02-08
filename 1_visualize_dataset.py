# 1_visualize_dataset.py

import Utils

from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box

import os
import pathlib
import numpy as np
import cv2
import copy
from typing import List
import plotly.graph_objects as PlotlyGraphObjects

LYFT_TRAIN_DATASET_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'train')

SHOW_PLOTLY_MOUSEOVERS = False

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    # load training data
    print('\n' + 'loading training data . . . ' + '\n')
    level5data = LyftDataset(data_path=LYFT_TRAIN_DATASET_LOC, json_path=os.path.join(LYFT_TRAIN_DATASET_LOC, 'data'), verbose=True)

    frameId: str = level5data.scene[0]['first_sample_token']
    sampleData: dict = level5data.get('sample', frameId)
    lidarTopId: str = sampleData['data']['LIDAR_TOP']
    lidarFilePathObj: pathlib.Path = level5data.get_sample_data_path(lidarTopId)
    lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePathObj)

    # intensity is always 100, so remove it
    lidarPoints: np.ndarray = lidarPointCloud.points[:3, :]

    lidarTopData: dict = level5data.get('sample_data', lidarTopId)
    gndTrBoxes: List[Box] = level5data.get_boxes(lidarTopId)

    ### 2D visualization ######################################################

    classes = ['car']
    tdvImageShape = (500, 500, 3)
    lidarRange = (200, 200, 4.5)
    zOffset = -2.0

    frameImage, _ = Utils.makeTdvImagesForFrame(level5data, frameId, classes, tdvImageShape, lidarRange, zOffset)

    egoPoseData: dict = level5data.get('ego_pose', lidarTopData['ego_pose_token'])

    gndTrBoxesCopy: List[Box] = copy.deepcopy(gndTrBoxes)

    Utils.moveBoxesFromWorldSpaceToCarSpace(gndTrBoxesCopy, egoPoseData)

    Utils.drawBoxesOnTdvImage(frameImage, tdvImageShape, gndTrBoxesCopy, (255, 255, 255), 1, lidarRange, zOffset)

    cv2.imwrite('0.png', frameImage)

    ### 3D visualization ######################################################

    s3dPoints = PlotlyGraphObjects.Scatter3d(x=lidarPoints[0], y=lidarPoints[1], z=lidarPoints[2], mode='markers', marker={'size': 1})

    # 3 separate lists for the x, y, and z components of each line
    xLines = []
    yLines = []
    zLines = []
    for box in gndTrBoxes:

        box = Utils.moveBoxFromWorldSpaceToSensorSpace(level5data, box, lidarTopData)

        corners = box.corners()

        # see here for documentation of Box:
        # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/data_classes.py#L622
        # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

        corners = corners.transpose()

        # 4 lines for front surface of box
        Utils.addLineToPlotlyLines(corners[0], corners[1], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[1], corners[2], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[2], corners[3], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[3], corners[0], xLines, yLines, zLines)

        # 4 lines between front points and read points
        Utils.addLineToPlotlyLines(corners[0], corners[4], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[1], corners[5], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[2], corners[6], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[3], corners[7], xLines, yLines, zLines)

        # 4 lines for rear surface of box
        Utils.addLineToPlotlyLines(corners[4], corners[7], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[5], corners[4], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[6], corners[5], xLines, yLines, zLines)
        Utils.addLineToPlotlyLines(corners[7], corners[6], xLines, yLines, zLines)

    # end for

    s3dGndTrBoxLines = PlotlyGraphObjects.Scatter3d(x=xLines, y=yLines, z=zLines, mode='lines')

    # make and show a plotly Figure object
    plotlyFig = PlotlyGraphObjects.Figure(data=[s3dPoints, s3dGndTrBoxLines])
    plotlyFig.update_layout(scene_aspectmode='data')

    if not SHOW_PLOTLY_MOUSEOVERS:
        plotlyFig.update_layout(hovermode=False)
        plotlyFig.update_layout(scene=dict(xaxis_showspikes=False,
                                           yaxis_showspikes=False,
                                           zaxis_showspikes=False))
    # end if

    plotlyFig.show()

# end function

if __name__ == '__main__':
    main()



