## Downloading images
You get a TIFF per band requested
```shell
python HLS_SuPER.py -roi ~/data/remote-sensing-comparison/test-area/test_area.geojson  -start 2024-09-01 -end 2024-11-10 -bands BLUE,GREEN,RED,NIR1,SWIR1,SWIR2,FMASK -qf True -dir  ~/data/remote-sensing-comparison/test-area
```

## Upsampling the lulc to 30x30m

1. Just do it to the LULC layer 
2. Grass -> raster -> r.resamp.stats
3. Dialog entries 
   4. Input is the LULC
   5. aggregation method is mode
   6. clear the quantile box
   7. GIS region extent is the entire study area
   8. cellsize is 30
9. This produces a 30m raster of the lulc BUT it is no longer aligned
10. Now run Raster tools -> Align Rasters
11. Dialog
    1. Input is the temp raster we made above
        1. Configure raster
        14. name is the lulc name with -align on the end
        15. nearest neighbor
    2. reference layer is one of the images
3. THe resulting image is the image to be used in analysis

Finished with the test area