import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio

'''
This should be run after preprocess_data.py
It takes the numpy arrays and creates a dataset that is easier to bring into fiftyone (and to visualize)

This is the best format for segmenation that matches 51
https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#imagesegmentationdirectory-import


'''