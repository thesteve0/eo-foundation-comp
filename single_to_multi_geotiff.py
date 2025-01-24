import os
import rasterio
import numpy as np
import fiftyone as fo
from rasterio.transform import xy

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


Create a dict that maps band names to index for TIFF
Same for the different PNGs we want
Make sure NOT to include FMASK for now

iterate through all the files in the directory
sorted(glob.glob('*.tif')) to insure they are in alphabetical order - needed below

This is the sequence that begins teh definition of images to stack
HLS.S30.T10SEG.2024314 - key in a dict - so chop this from the file first, check to see if there is already an entry into the dict if not time to make a dict that is a stack of bands
Create a dict of the different bands - key = layer name (red, green...) value is the dst from the "with rasterio.open() as dst


Once we have iterated through all the files on disk the unique file name dict should be done.
Create the fiftyone dataset
Now iterate through this and write out the files. Each one should be
The 6 band tiff naming is HLS.S30.T10SEG.2024314.tiff
The pngs naming is HLS.S30.T10SEG.2024314-name.png where name = rgb or fcir or swir...
The fiftyone sample, it will need a geo label - see the source for how to grab it from the TIFF
Do I add the clipped LULC mask here?

They should all be in the same directory. I think we should probably also make the "full_remote_sensed_images" 51 dataset at this time.
'''


input_dir: str = './images'
output_dir: str = './multi-band-images'

# Bands in order: BLUE, GREEN, RED, NIR, SWIR1, SWIR2
BAND_ORDER: list[str] = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2']


def process_multiband_images() -> None:
    image_groups: dict[str, list[str]] = {}
    for filename in os.listdir(input_dir):
        base_name = filename[:22]
        if base_name not in image_groups:
            image_groups[base_name] = []
        image_groups[base_name].append(filename)

    os.makedirs(output_dir, exist_ok=True)

    # Create FiftyOne dataset
    dataset = fo.Dataset("rs_full_images")

    for base_name, band_files in image_groups.items():
        sorted_files = sorted(band_files, key=lambda f: BAND_ORDER.index(f.split('.')[-2]))

        multiband_path = os.path.join(output_dir, f'{base_name}.subset.tif')
        merge_bands(sorted_files, multiband_path)

        # Create PNGs
        png_configs = [
            ('rgb', [1, 2, 3])
        ]

        # Prepare sample metadata
        sample_metadata = {}
        for png_suffix, bands in png_configs:
            png_path = multiband_path.replace('.subset.tif', f'.subset.{png_suffix}.png')
            create_png(multiband_path, png_suffix, bands)
            sample_metadata[f"{png_suffix}_display"] = png_path

        # Get geospatial information
        with rasterio.open(multiband_path) as src:
            # Get corner coordinates
            transform = src.transform
            height, width = src.height, src.width

            # Get center coordinates
            center_lon, center_lat = xy(transform, height // 2, width // 2)

            # Create FiftyOne sample
            # Create FiftyOne sample with geolocation
            sample = fo.Sample(
                filepath=multiband_path,
                location=fo.GeoLocation(point=[center_lon, center_lat])
            )

            # Add PNG display paths
            for key, path in sample_metadata.items():
                sample[key] = path

            dataset.add_sample(sample)

    dataset.save()


def merge_bands(band_files: list[str], output_path: str) -> None:
    bands = []
    with rasterio.open(os.path.join(input_dir, band_files[0])) as first_src:
        # Use first band's metadata as template
        profile = first_src.profile.copy()

        # Read bands for stacking
        for band_file in band_files:
            with rasterio.open(os.path.join(input_dir, band_file)) as src:
                bands.append(src.read(1))

    # Stack bands
    stacked_bands = np.stack(bands)

    # Update profile for multiband image
    profile.update(
        count=len(bands),
        dtype=stacked_bands.dtype
    )

    # Write multiband image with original geographic metadata
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(stacked_bands)


def create_png(multiband_path: str, png_suffix: str, bands_to_use: list[int]) -> None:
    with rasterio.open(multiband_path) as src:
        # Select specified bands
        selected_bands = src.read(bands_to_use)

        # Normalize to 0-255 range
        normalized = (((selected_bands - selected_bands.min()) / (
                    selected_bands.max() - selected_bands.min())) * 255).astype(np.uint8)

        png_path = multiband_path.replace('.subset.tif', f'.subset.{png_suffix}.png')

        # Write PNG
        with rasterio.open(png_path, 'w', driver='PNG', width=src.width, height=src.height,
                           count=len(bands_to_use), dtype=np.uint8) as png:
            png.write(normalized)

if __name__ == '__main__':
    process_multiband_images()