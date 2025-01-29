r"""
Chesapeake CVPR Data Processing Script modified by S. Pousty for Santa Cruz LULC work
======================================

Original is here
https://github.com/Clay-foundation/model/blob/main/finetune/segment/preprocess_data.py

I am modifying the notes below to fit this current project.

This script processes GeoTIFF files from NASA HLS and an LULC TIFF from  Conservation Lands Network.
Using FULL CLN 2.0 GIS DATABASE (Version 2.0.1)
https://www.bayarealands.org/maps-data/

Notes:
------
This only creates serialized numpy representations of the chip and not actual images


We will create chips of size `224 x 224` to feed them to the model
   Example:
   python preprocess_data.py data/cvpr/files data/cvpr/ny 224
"""

import os
import sys
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.windows import Window, from_bounds
import fiftyone as fo

DATASET_NAME = "rs_chip_images"


def get_corresponding_window(
        ref_bounds: rio.coords.BoundingBox,
        ref_transform: rio.Affine,
        target_src: rio.DatasetReader,
        chip_size: int,
        chip_x: int,
        chip_y: int
) -> Window:
    """
    Calculate the window in target_src that corresponds to a chip position in reference image.

    Args:
        ref_bounds: Bounds of the reference image
        ref_transform: Transform of the reference image
        target_src: The target raster dataset
        chip_size: Size of chip in pixels
        chip_x: Chip index in x direction
        chip_y: Chip index in y direction

    Returns:
        Window: A rasterio Window object defining the pixel coordinates in target_src
    """
    # Get the real-world coordinates of the chip in the reference image
    minx = ref_bounds.left + (chip_x * chip_size * ref_transform.a)
    maxx = minx + (chip_size * ref_transform.a)
    maxy = ref_bounds.top - (chip_y * chip_size * abs(ref_transform.e))
    miny = maxy - (chip_size * abs(ref_transform.e))

    # Convert these coordinates to pixel coordinates in the target image
    # Window.from_bounds converts geographic bounds to pixel coordinates
    window = from_bounds(
        left=minx,
        bottom=miny,
        right=maxx,
        top=maxy,
        transform=target_src.transform
    )

    # Round the window coordinates to ensure integer pixel addressing
    return Window(
        col_off=round(window.col_off),
        row_off=round(window.row_off),
        width=round(window.width),
        height=round(window.height)
    )

def read_and_chip(file_path, chip_size, output_dir, ground_truth_output_dir, ground_truth_file, fiftyone_dataset):
    """
    Reads a GeoTIFF file, creates chips of specified size, and saves them as
    numpy arrays.

    Args:
        file_path (str or Path): Path to the GeoTIFF file.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ground_truth_output_dir, exist_ok=True)

    with rio.open(file_path) as src:
        data = src.read()
        ref_bounds = src.bounds
        profile = src.profile.copy()
        profile.update(
            count=src.count,
            width=chip_size,
            height=chip_size,
            dtype="float32",
            driver='GTiff',
            compress='lzw'  # Add compression to ensure proper writing
        )

        num_chips_x = src.width // chip_size
        num_chips_y = src.height // chip_size

        chip_number = 0
        for chip_x in range(num_chips_x):
            for chip_y in range(num_chips_y):
                output_base = os.path.join(output_dir, f"{Path(file_path).stem}_chip_{chip_number}")
                grnd_truth_output_file_base = os.path.join(ground_truth_output_dir, f"{Path(file_path).stem}_lulc_chip_{chip_number}")

                # For numpy slicing and
                x1, y1 = chip_x * chip_size, chip_y * chip_size
                x2, y2 = x1 + chip_size, y1 + chip_size
                chip = data[:, y1:y2, x1:x2]

############################# LULC Ground truth image creation ###############

                # Now calculate the coordinates for this chip to cut from LULC
                # Calculate the window coordinates
                x_offset = chip_x * chip_size
                y_offset = chip_y * chip_size

                # Handle edge cases where tile would exceed image bounds
                width = min(chip_size, src.width - x_offset)
                height = min(chip_size, src.height - y_offset)

                # Create the window
                window = Window(x_offset, y_offset, width, height)


                # Read the lulc data in this window
                with rio.open(ground_truth_file) as grnd_truth:
                    if src.crs != grnd_truth.crs:
                        raise ValueError(f"CRS must match between reference and ground truth images. "
                                         f"Got {src.crs} and {grnd_truth.crs}")

                    grnd_truth_profile = grnd_truth.profile.copy()
                    gt_window = get_corresponding_window(
                        ref_bounds=ref_bounds,
                        ref_transform=src.transform,
                        target_src=grnd_truth,
                        chip_size=chip_size,
                        chip_x=chip_x,
                        chip_y=chip_y
                    )


                    grnd_truth_data = grnd_truth.read(window=gt_window)

                    np.save(grnd_truth_output_file_base + ".npy", grnd_truth_data)

                    # Calculate new transform for this tile
                    grnd_truth_transform = rio.windows.transform(gt_window, grnd_truth.transform)
                    grnd_truth_profile.update({
                        'height': height,
                        'width': width,
                        'transform': grnd_truth_transform
                    })
                    with rio.open(grnd_truth_output_file_base + ".tif", 'w', **grnd_truth_profile) as dst:
                        dst.write(grnd_truth_data)
                        dst.close()

                    #now the png
                    grnd_truth_profile.update(
                        driver = 'PNG',
                        crs = None,
                        transform = None,
                        dtype = np.uint8,
                        count = 1,
                        compress = None
                    )
                    with rio.open(grnd_truth_output_file_base + ".png", 'w', **grnd_truth_profile) as dst:
                        dst.write(grnd_truth_data)
                        dst.close()





                # when we load a numpy array and it is all NaN we should delete all the files with that name and remove the sample
                if np.isnan(chip).all():
                    print("empty numpy array - skip creating anything:: " + output_base )
                else:
                    chip_path = output_base + f".npy"
                    tif_path = output_base + f".tif"

                    created_pngs = {}
                    rgb_path = output_base + ".rgb.png"
                    fcir_path = output_base + ".fcir.png"
                    sveg_path = output_base + ".sveg.png"

                    created_pngs["rgb_display"] = rgb_path
                    created_pngs["fcir_display"] = fcir_path
                    created_pngs["sveg_display"] = sveg_path

                    # in the future I will add geotiffs but for right now, geotiffs are not needed
                    # Here we are going to create a tiff, some PNGs, and a sample, with this np aray as a segmentation label

                    # This np.save is the original which just saves a np array to disk
                    np.save(chip_path, chip)

                    profile.update(
                        transform = rio.windows.transform(window, src.transform),
                    )

                    # Now tiff
                    with rio.open(tif_path, "w", **profile) as chip_tiff:
                        chip_tiff.write(chip)
                        chip_tiff.close()
                    # Verify file was written
                    if not os.path.exists(tif_path) or os.path.getsize(tif_path) < 1000:
                        raise ValueError(f"TIF file not written correctly: {tif_path}")


                    if fiftyone_dataset is not None:
                        # Make the 3 pngs - it is more likely for these to fail because
                        # - if one of the bands required is missing this will fail
                        # - a chip could actually have no data in it - these should not be fed into the model
                        try:
                            create_png(tif_path, 'rgb', [3, 2, 1])  # Adjusted band indices to [3,2,1] for RGB
                        except Exception as e:
                            print(f"Error creating RGB for {tif_path}: {str(e)}")
                            continue
                        try:
                            create_png(tif_path, 'fcir', [4, 3, 2])
                        except Exception as e:
                            print(f"Error creating fcir for {tif_path}: {str(e)}")
                            continue
                        try:
                            create_png(tif_path, 'sveg', [5, 4, 3])
                        except Exception as e:
                            print(f"Error creating sveg for {tif_path}: {str(e)}")
                            continue


                        # Now populate the dataset
                        create_and_add_sample(tiff_path=tif_path,png_dict=created_pngs, chip_path=chip_path, dataset=fiftyone_dataset)

                chip_number += 1


def create_and_add_sample(tiff_path: str, png_dict: dict, chip_path: str, dataset: fo.Dataset) -> None:
    sample = fo.Sample(filepath=tiff_path)
    sample["rgb_display"] = png_dict["rgb_display"]
    sample["fcir_display"] = png_dict["fcir_display"]
    sample["sveg_display"] = png_dict["sveg_display"]
    sample["chip_path"] = chip_path
    dataset.add_sample(sample)



def create_png(multiband_path: str, png_suffix: str, bands_to_use: list[int]) -> None:
    # Ensure file exists and has non-zero size
    if not os.path.exists(multiband_path) or os.path.getsize(multiband_path) == 0:
        raise ValueError(f"Invalid TIF file: {multiband_path}")

    with rio.open(multiband_path) as src:
        # Verify we can read the bands
        selected_bands = []
        for band_idx in bands_to_use:
            band_data = src.read(band_idx)
            if band_data is None or np.all(np.isnan(band_data)):
                raise ValueError(f"Invalid band data for band {band_idx}")
            selected_bands.append(band_data)

        selected_bands = np.array(selected_bands)

        # Normalize each band separately
        normalized = np.zeros_like(selected_bands, dtype=np.uint8)
        for i in range(len(bands_to_use)):
            band = selected_bands[i]
            if np.all(np.isnan(band)):
                raise ValueError(f"Band {i} contains all NaN values")
            p2, p98 = np.percentile(band[~np.isnan(band)], (2, 98))
            normalized[i] = np.clip(((band - p2) / (p98 - p2) * 255), 0, 255).astype(np.uint8)

        png_path = multiband_path.replace('.tif', f'.{png_suffix}.png')

        profile = src.profile.copy()
        profile.update(
            driver='PNG',
            crs=None,
            transform=None,
            dtype=np.uint8,
            count=len(bands_to_use),
            compress=None
        )

        with rio.open(png_path, 'w', **profile) as png:
            png.write(normalized)
            png.close()

def process_files_with_dataset(file_paths, output_dir, ground_truth_output_dir, ground_truth_file, chip_size, dataset_name):
    dataset = fo.Dataset(dataset_name, overwrite=True, persistent=True)
    process_files(file_paths, output_dir, ground_truth_output_dir, ground_truth_file, chip_size, dataset)
    dataset.app_config.media_fields = ["filepath", "rgb_display", "fcir_display", "sveg_display"]
    dataset.app_config.grid_media_field = "rgb_display"
    dataset.save()


def process_files(file_paths, output_dir, ground_truth_output_dir, ground_truth_file, chip_size, dataset=None):
    """
    Processes a list of files, creating chips and saving them.

    Args:
        file_paths (list of Path): List of paths to the GeoTIFF files.
        output_dir (str or Path): Directory to save the chips.
        chip_size (int): Size of the square chips.
    """
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        read_and_chip(file_path, chip_size, output_dir, ground_truth_output_dir, ground_truth_file, dataset)


def main():
    """
    Main function to process files and create chips.
    Expects three command line arguments:
        - data_dir: Directory containing the input GeoTIFF files.
        - output_dir: Directory to save the output chips.
        - chip_size: Size of the square chips.
    """
    if len(sys.argv) != 4:  # noqa: PLR2004
        print("Usage: python preprocess_data.py <data_dir> <output_dir> <chip_size>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    chip_size = int(sys.argv[3])

    # I am going to process all the files and then split into train and validation using fiftyone
    # train_image_paths = list((data_dir / "train").glob("*_naip-new.tif"))
    # val_image_paths = list((data_dir / "val").glob("*_naip-new.tif"))
    # train_label_paths = list((data_dir / "train").glob("*_lc.tif"))
    # val_label_paths = list((data_dir / "val").glob("*_lc.tif"))

    image_paths = list((data_dir / "multi-band-images").glob("*.tif"))
    ground_truth_file = os.path.join(data_dir, "ground-truth/lulc_30m_align.tif")


    # This just needs 2 entries, one for labels and one for images
    # process_files(ground_truth_paths, output_dir / "ground-truth-chips", chip_size)
    process_files_with_dataset(image_paths,
                               output_dir / "multi-band-image-chips",
                               output_dir / "ground-truth-chips",
                               ground_truth_file,
                               chip_size,
                               DATASET_NAME)


if __name__ == "__main__":
    main()
