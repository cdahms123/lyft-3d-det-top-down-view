# 3_visualize_val.py

import Utils

from lyft_dataset_sdk.lyftdataset import LyftDataset, Box
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud

import os
import pathlib
import pandas as pd
import cv2
import numpy as np
import copy
import plotly.graph_objects as PlotlyGraphObjects
from typing import List

LYFT_TRAIN_DATASET_LOC = os.path.join(os.path.expanduser('~'), 'LyftObjDetDataset', 'train')
CSV_LOC = os.path.join(os.getcwd(), 'csv')

def main():
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    df = pd.read_csv(os.path.join(CSV_LOC, 'my_val.csv'))

    print(df)

    # load training data
    print('\n' + 'loading training data . . . ' + '\n')
    level5data = LyftDataset(data_path=LYFT_TRAIN_DATASET_LOC, json_path=os.path.join(LYFT_TRAIN_DATASET_LOC, 'data'), verbose=True)

    print('\n' + 'building boxes lists . . . ' + '\n')
    allPredBoxes: List[List[Box]] = []
    for idx in range(len(df)):
        predBoxes = Utils.getPredBoxes(df, idx)
        allPredBoxes.append(predBoxes)
    # end function

    idx = 850  # change this to visualize other samples

    # get the frame ID for the current frame
    frameId = df.iloc[idx]['Id']
    # get the pred boxes for the current frame
    predBoxes = allPredBoxes[idx]

    sampleData: dict = level5data.get('sample', frameId)
    lidarTopId: str = sampleData['data']['LIDAR_TOP']
    lidarFilePathObj: pathlib.Path = level5data.get_sample_data_path(lidarTopId)
    lidarPointCloud: LidarPointCloud = LidarPointCloud.from_file(lidarFilePathObj)

    # intensity is always 100, so remove it
    lidarPoints: np.ndarray = lidarPointCloud.points[:3, :]

    lidarTopData: dict = level5data.get('sample_data', lidarTopId)
    gndTrBoxes: List[Box] = level5data.get_boxes(lidarTopId)

    ### 2D visualization ############################################

    classes = ['car']
    tdvImageShape = (500, 500, 3)
    lidarRange = (200, 200, 4.5)
    zOffset = -2.0

    frameImage, _ = Utils.makeTdvImagesForFrame(level5data, frameId, classes, tdvImageShape, lidarRange, zOffset)

    egoPoseData: dict = level5data.get('ego_pose', lidarTopData['ego_pose_token'])

    predBoxesCopy = copy.deepcopy(predBoxes)
    Utils.moveBoxesFromWorldSpaceToCarSpace(predBoxesCopy, egoPoseData)
    Utils.drawBoxesOnTdvImage(frameImage, tdvImageShape, predBoxesCopy, (0, 0, 255), 1, lidarRange, zOffset)

    gndTrBoxesCopy = copy.deepcopy(gndTrBoxes)
    Utils.moveBoxesFromWorldSpaceToCarSpace(gndTrBoxesCopy, egoPoseData)
    Utils.drawBoxesOnTdvImage(frameImage, tdvImageShape, gndTrBoxesCopy, (255, 0, 0), 1, lidarRange, zOffset)

    cv2.imwrite('2.png', frameImage)

    ### 3D visualization ############################################

    s3dPoints = PlotlyGraphObjects.Scatter3d(x=lidarPoints[0], y=lidarPoints[1], z=lidarPoints[2], mode='markers', marker={'size': 1})

    # 3 separate lists for the x, y, and z components of each line
    predXLines = []
    predYLines = []
    predZLines = []
    for predBox in predBoxes:
        predBox = Utils.moveBoxFromWorldSpaceToSensorSpace(level5data, predBox, lidarTopData)

        corners = predBox.corners()

        # see here for documentation of Box:
        # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/data_classes.py#L622
        # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

        corners = corners.transpose()

        # 4 lines for front surface of box
        Utils.addLineToPlotlyLines(corners[0], corners[1], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[1], corners[2], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[2], corners[3], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[3], corners[0], predXLines, predYLines, predZLines)

        # 4 lines between front points and read points
        Utils.addLineToPlotlyLines(corners[0], corners[4], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[1], corners[5], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[2], corners[6], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[3], corners[7], predXLines, predYLines, predZLines)

        # 4 lines for rear surface of box
        Utils.addLineToPlotlyLines(corners[4], corners[7], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[5], corners[4], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[6], corners[5], predXLines, predYLines, predZLines)
        Utils.addLineToPlotlyLines(corners[7], corners[6], predXLines, predYLines, predZLines)

    # end for

    s3dPredBoxLines = PlotlyGraphObjects.Scatter3d(x=predXLines, y=predYLines, z=predZLines, mode='lines')

    # 3 separate lists for the x, y, and z components of each line
    gndTrXLines = []
    gndTrYLines = []
    gndTrZLines = []
    for gndTrBox in gndTrBoxes:
        gndTrBox = Utils.moveBoxFromWorldSpaceToSensorSpace(level5data, gndTrBox, lidarTopData)

        corners = gndTrBox.corners()

        # see here for documentation of Box:
        # https://github.com/lyft/nuscenes-devkit/blob/master/lyft_dataset_sdk/utils/data_classes.py#L622
        # when getting corners, the first 4 corners are the ones facing forward, the last 4 are the ones facing rearwards

        corners = corners.transpose()

        # 4 lines for front surface of box
        Utils.addLineToPlotlyLines(corners[0], corners[1], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[1], corners[2], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[2], corners[3], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[3], corners[0], gndTrXLines, gndTrYLines, gndTrZLines)

        # 4 lines between front points and read points
        Utils.addLineToPlotlyLines(corners[0], corners[4], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[1], corners[5], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[2], corners[6], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[3], corners[7], gndTrXLines, gndTrYLines, gndTrZLines)

        # 4 lines for rear surface of box
        Utils.addLineToPlotlyLines(corners[4], corners[7], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[5], corners[4], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[6], corners[5], gndTrXLines, gndTrYLines, gndTrZLines)
        Utils.addLineToPlotlyLines(corners[7], corners[6], gndTrXLines, gndTrYLines, gndTrZLines)

    # end for

    s3dGndTrBoxLines = PlotlyGraphObjects.Scatter3d(x=gndTrXLines, y=gndTrYLines, z=gndTrZLines, mode='lines')

    # make and show a plotly Figure object
    plotlyFig = PlotlyGraphObjects.Figure(data=[s3dPoints, s3dPredBoxLines, s3dGndTrBoxLines])
    plotlyFig.update_layout(scene_aspectmode='data')
    plotlyFig.show()
# end function

if __name__ == '__main__':
    main()

















