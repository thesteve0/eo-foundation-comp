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
import fiftyone as fo

DATASET_NAME = "rs_chip_images"


def read_and_chip(file_path, chip_size, output_dir, fiftyone_dataset):
    """
    Reads a GeoTIFF file, creates chips of specified size, and saves them as
    numpy arrays.

    Args:
        file_path (str or Path): Path to the GeoTIFF file.
        chip_size (int): Size of the square chips.
        output_dir (str or Path): Directory to save the chips.
    """
    os.makedirs(output_dir, exist_ok=True)

    with rio.open(file_path) as src:
        data = src.read()
        profile = src.profile.copy()
        profile.update(
            crs=None,
            transform=None,
            count=src.count,
            width=chip_size,
            height=chip_size,
            dtype="float32",
            driver='GTiff',
            compress='lzw'  # Add compression to ensure proper writing
        )

        n_chips_x = src.width // chip_size
        n_chips_y = src.height // chip_size

        chip_number = 0
        for i in range(n_chips_x):
            for j in range(n_chips_y):
                x1, y1 = i * chip_size, j * chip_size
                x2, y2 = x1 + chip_size, y1 + chip_size

                chip = data[:, y1:y2, x1:x2]
                output_base = os.path.join( output_dir,f"{Path(file_path).stem}_chip_{chip_number}")
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

                # Now tiff
                with rio.open(tif_path, "w", **profile) as chip_tiff:
                    chip_tiff.write(chip)
                    chip_tiff.close()
                # Verify file was written
                if not os.path.exists(tif_path) or os.path.getsize(tif_path) < 1000:
                    raise ValueError(f"TIF file not written correctly: {tif_path}")


                if fiftyone_dataset is not None:
                    # Make the 3 pngs
                    try:
                        create_png(tif_path, 'rgb', [3, 2, 1])  # Adjusted band indices to [3,2,1] for RGB
                        create_png(tif_path, 'fcir', [4, 3, 2])
                        create_png(tif_path, 'sveg', [5, 4, 3])
                    except Exception as e:
                        print(f"Error creating PNG for {tif_path}: {str(e)}")
                        continue

                    # Now populate the dataset
                    create_and_add_sample(tiff_path=tif_path,png_dict=created_pngs, dataset=fiftyone_dataset)

                chip_number += 1


def create_and_add_sample(tiff_path: str, png_dict: dict, dataset: fo.Dataset) -> None:
    sample = fo.Sample(filepath=tiff_path)
    sample["rgb_display"] = png_dict["rgb_display"]
    sample["fcir_display"] = png_dict["fcir_display"]
    sample["sveg_display"] = png_dict["sveg_display"]
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

def process_files_with_dataset(file_paths, output_dir, chip_size, dataset_name):
    dataset = fo.Dataset(dataset_name, overwrite=True, persistent=True)
    process_files(file_paths, output_dir, chip_size, dataset)
    dataset.app_config.media_fields = ["filepath", "rgb_display", "fcir_display", "sveg_display"]
    dataset.app_config.grid_media_field = "rgb_display"
    dataset.save()


def process_files(file_paths, output_dir, chip_size, dataset=None):
    """
    Processes a list of files, creating chips and saving them.

    Args:
        file_paths (list of Path): List of paths to the GeoTIFF files.
        output_dir (str or Path): Directory to save the chips.
        chip_size (int): Size of the square chips.
    """
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        read_and_chip(file_path, chip_size, output_dir, dataset)


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
    ground_truth_paths = list((data_dir / "ground-truth").glob("*.tif"))


    # This just needs 2 entries, one for labels and one for images
    process_files_with_dataset(image_paths, output_dir / "multi-band-image-chips", chip_size, DATASET_NAME)
    process_files(ground_truth_paths, output_dir / "ground-truth-chips", chip_size)


if __name__ == "__main__":
    main()
