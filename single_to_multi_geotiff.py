import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio

'''
This script is specifically for dealing with HLS data which comes back one band at a time. It is also geared towards just
the Landsat data
https://lpdaac.usgs.gov/data/get-started-data/collection-overview/missions/harmonized-landsat-sentinel-2-hls-overview/#hls-naming-conventions

filenames look like this:
HLS.S30.T10SEG.2024314T185611.v2.0.SWIR2.subset.tif

HLS.S30 or L30 is just sentinel vs landsat
* T10SEG is the actual scene tile ID - t + 5 character ID
* The next 7 digits are year and day of year so jan 1 2024 would look like 2024001
* The T and digits afterwards is the Time of Acquisition (HHMMSS)

To create a multiband geotiff
https://gis.stackexchange.com/questions/464604/write-three-band-rgb-to-file
https://medium.com/@saurabhkr.gupta/layer-stacking-of-raster-files-through-python-and-google-colab-of-sentinel-2-data-723f910af886

Then to export the PNGs for the band combinations
https://gis.stackexchange.com/questions/466653/how-to-write-png-file-from-a-raster-using-rasterio
https://mapscaping.com/converting-a-geotiff-to-png/

May need to reshape the png
https://rasterio.readthedocs.io/en/stable/topics/image_processing.html
'''

# Create a dict that maps band names to index for TIFF
# Same for the different PNGs we want
# Make sure NOT to include FMASK for now

# iterate through all the files in the directory
# sorted(glob.glob('*.tif')) to insure they are in alphabetical order - needed below

# This is the sequence that begins teh definition of images to stack
# HLS.S30.T10SEG.2024314 - key in a dict - so chop this from the file first, check to see if there is already an entry into the dict if not time to make a dict that is a stack of bands
# Create a dict of the different bands - key = layer name (red, green...) value is the dst from the "with rasterio.open() as dst


# Once we have iterated through all the files on disk the unique file name dict should be done.
# Create the fiftyone dataset
# Now iterate through this and write out the files. Each one should be
# The 6 band tiff naming is HLS.S30.T10SEG.2024314.tiff
# The pngs naming is HLS.S30.T10SEG.2024314-name.png where name = rgb or fcir or swir...
# The fiftyone sample, it will need a geo label - see the source for how to grab it from the TIFF
# Do I add the clipped LULC mask here?

# They should all be in the same directory. I think we should probably also make the "full_remote_sensed_images" 51 dataset at this time.