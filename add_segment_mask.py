import re
import fiftyone as fo
import numpy as np
import rasterio as rio

'''
To set up a id to name mapping for the segmentation masks
https://docs.voxel51.com/user_guide/using_datasets.html#storing-mask-targets

To make a custom color scheme 
https://docs.voxel51.com/user_guide/app.html#color-schemes-in-python
https://docs.voxel51.com/api/fiftyone.core.odm.dataset.html#fiftyone.core.odm.dataset.ColorScheme

'''

DATASET_NAME = "rs_chip_images"
SEGMENT_MASK_PATH = "/home/spousty/data/remote-sensing-comparison/test-area/ground-truth-chips/test_area_lulc_30m_align_"

def load_dataset_and_add_segment_mask(dataset):
    if dataset.has_field("ground_truth"):
        dataset.delete_sample_field("ground_truth")

    dataset.set_values(
        "ground_truth",
        [fo.Segmentation() for _ in range(len(dataset))],
    )

    # This image
    # '/home/spousty/data/remote-sensing-comparison/test-area/multi-band-image-chips/HLS.L30.T10SEG.2024297.subset_chip_0.tif'
    # Should get this mask
    # /home/spousty/data/remote-sensing-comparison/test-area/ground-truth-chips/test_area_lulc_30m_align_chip_0.npy

    # Which means replace this:
    # /home/spousty/data/remote-sensing-comparison/test-area/multi-band-image-chips/HLS.L30.T10SEG.2024297.subset_
    # with
    # /home/spousty/data/remote-sensing-comparison/test-area/ground-truth-chips/test_area_lulc_30m_align_
    # Insert the chip_X and then add a .npy on the end

    # Get all the paths for the tiffs
    segmentation_arrays = []
    paths = dataset.values("filepath")
    for path in paths:
        chip_number = re.search(r'chip_(\d+)', path).group(0)
        lulc_np = SEGMENT_MASK_PATH + chip_number + ".npy"
        lulc_png = SEGMENT_MASK_PATH + chip_number + ".png"

        image_as_np = np.load(lulc_np).astype("uint8")

        with rio.open(lulc_png, 'w', driver='PNG', dtype=np.uint8, count=1, width=image_as_np.shape[1], height=image_as_np.shape[1]) as png:
            png.write(image_as_np)
            png.close()
        segmentation_arrays.append(lulc_png)

    dataset.set_values("ground_truth.mask_path", segmentation_arrays)



if __name__ == '__main__':
    print("Starting")
    dataset = fo.load_dataset(DATASET_NAME)

    load_dataset_and_add_segment_mask(dataset)

    print("finished")