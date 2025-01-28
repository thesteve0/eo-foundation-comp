import os
import rasterio
import numpy as np
import fiftyone as fo
from rasterio.transform import xy
from pyproj import Transformer

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


input_dir: str = '/home/spousty/data/remote-sensing-comparison/images'
output_dir: str = '/home/spousty/data/remote-sensing-comparison/multi-band-images'

# Bands in order: BLUE, GREEN, RED, NIR1, SWIR1, SWIR2
BAND_ORDER: list[str] = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2']


def merge_bands(band_files: list[str], output_path: str) -> None:
    bands = []
    with rasterio.open(os.path.join(input_dir, band_files[0])) as first_src:
        profile = first_src.profile.copy()

        for band_file in band_files:
            with rasterio.open(os.path.join(input_dir, band_file)) as src:
                bands.append(src.read(1))

    stacked_bands = np.stack(bands)

    profile.update(
        count=len(bands),
        dtype=stacked_bands.dtype,
        driver='GTiff',
        compress='lzw'  # Add compression to ensure proper writing
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(stacked_bands)
        dst.close()  # Explicitly close


def create_png(multiband_path: str, png_suffix: str, bands_to_use: list[int]) -> None:
    # Ensure file exists and has non-zero size
    if not os.path.exists(multiband_path) or os.path.getsize(multiband_path) == 0:
        raise ValueError(f"Invalid TIF file: {multiband_path}")

    with rasterio.open(multiband_path) as src:
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

        png_path = multiband_path.replace('.subset.tif', f'.subset.{png_suffix}.png')

        profile = src.profile.copy()
        profile.update(
            driver='PNG',
            dtype=np.uint8,
            count=len(bands_to_use),
            compress=None
        )

        with rasterio.open(png_path, 'w', **profile) as png:
            png.write(normalized)
            png.close()


def process_multiband_images() -> None:
    image_groups: dict[str, list[str]] = {}
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') and not filename.__contains__("FMASK"):
            base_name = filename[:22]
            if base_name not in image_groups:
                image_groups[base_name] = []
            image_groups[base_name].append(filename)

    os.makedirs(output_dir, exist_ok=True)
    dataset = fo.Dataset("rs_full_images", overwrite=True, persistent=True)

    for base_name, band_files in image_groups.items():
        sorted_files = sorted(band_files, key=lambda f: BAND_ORDER.index(f.split('.')[-3]))
        multiband_path = os.path.join(output_dir, f'{base_name}.subset.tif')

        # Create and write the multiband tiff
        merge_bands(sorted_files, multiband_path)

        # Verify file was written
        if not os.path.exists(multiband_path) or os.path.getsize(multiband_path) < 1000:
            raise ValueError(f"TIF file not written correctly: {multiband_path}")

        # Create the PNG
        try:
            create_png(multiband_path, 'rgb', [3, 2, 1])  # Adjusted band indices to [3,2,1] for RGB
            create_png(multiband_path, 'fcir', [4, 3, 2])
            create_png(multiband_path, 'sveg', [5, 4, 3])


        except Exception as e:
            print(f"Error creating PNG for {base_name}: {str(e)}")
            continue

        # Get coordinates for the sample and project to 4326 (DD, wgs84)
        with rasterio.open(multiband_path) as src:
            transformer = Transformer.from_crs("EPSG:32610", "EPSG:4326")
            transform = src.transform
            height, width = src.height, src.width
            center_lon, center_lat = xy(transform, height // 2, width // 2)
            center_lat, center_lon = transformer.transform(center_lat, center_lon)



        # Create FiftyOne sample
        sample = fo.Sample(
            filepath=multiband_path,
            location=fo.GeoLocation(point=[center_lon, center_lat])
        )
        sample["rgb_display"] = multiband_path.replace('.subset.tif', '.subset.rgb.png')
        sample["fcir_display"] = multiband_path.replace('.subset.tif', '.subset.fcir.png')
        sample["sveg_display"] = multiband_path.replace('.subset.tif', '.subset.sveg.png')
        dataset.add_sample(sample)

    dataset.app_config.media_fields = ["filepath", "rgb_display", "fcir_display", "sveg_display"]
    dataset.app_config.grid_media_field = "rgb_display"
    dataset.save()


if __name__ == '__main__':
    process_multiband_images()