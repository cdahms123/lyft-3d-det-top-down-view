### lyft-3d-det-top-down-view

#### Instructions to run

1) Install the Lyft object detection Python library
```
sudo -H pip3 install lyft_dataset_sdk
```

1) Go to https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles/data and download the dataset (scroll to the bottom then look for `Download All` on the left), then unzip it.  This will take time as the .zip is 91.1 GB and the unzipped content is 125.8 GB.

2) Rename the extracted directory to `lyft-obj-det-dataset` and move it to your home directory.  Next, inside `~/home/lyft-obj-det-dataset`, make a directory `train` and a directory `test`, then move the 4 directories that start `train_` into `train` and the 4 directories that start `test_` into `test`.  Finally, inside `train` remove the `train_` from the 4 directories, and inside `test` remove the `test_` from the 4 directories.  When complete your directory structure should be like:

```
~
|-- lyft-obj-det-dataset
     |-- test
          |-- data
          +-- images
          +-- lidar
          +-- maps
     |-- train
          |-- data
          +-- images
          +-- lidar
          +-- maps
     +-- sample_submission.csv
     +-- train.csv
```

3) Run the scripts 0-4 in order








