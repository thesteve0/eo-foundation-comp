# eo-foundation-comp
Code and simple doc comparing different foundational earth observation vision models

## ToDo
0. Install all the dependencies in the venv
1. Chip creation - can be used in both models.Chips the images AND the LULC using this code from Clay:
https://github.com/Clay-foundation/model/blob/main/finetune/segment/preprocess_data.py

2. Create the correct metadata file for Clay: https://github.com/Clay-foundation/model/blob/main/configs/segment_chesapeake.yaml
3. Import these into Fiftyone and they should have segmentation groundth truth labels on the original images

## Notes
### How many parameters and which model to compare
It looks like the Clay model is about 600M parameters
https://clay-foundation.github.io/model/release-notes/specification.html#model-architecture

Which means we should use the 600M parameter TL Prithvi
https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL

### Chipping
I have to build time stacks because not all of my subject area is covered with each pass of the satellite

Looks like Prithvi likes 224x224
https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/blob/main/README.md

and Clay should be pretty good with that as well.
https://clay-foundation.github.io/model/release-notes/specification.html#pre-training-and-usage

    Clay v1.5 was trained on 70 million globally distributed chips of size 156x256, 
    collected according to the land use/land cover (LULC) statistics of the globe.

In the example code they are interpolating images to 224 as well:
https://github.com/Clay-foundation/model/blob/c661ea44284bc229078050eebca81f7ef4a8b765/finetune/segment/chesapeake_model.py#L129


### Clay
Clay Discussion for chesapeake segmentation - use this cli but maybe not since I want to attach the FiftyOne predictions to them
https://clay-foundation.github.io/model/finetune/segment.html

Notebook for this
https://github.com/Clay-foundation/model/blob/main/finetune/segment/chesapeake_inference.ipynb

Full requirements for Clay
https://github.com/Clay-foundation/model/blob/main/environment.yml

### Prithvi for Multitemporal crop

Data discussion
https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification/blob/main/README.md

Notebook
https://github.com/NASA-IMPACT/Prithvi-EO-2.0/blob/main/examples/example_multitemporalcrop.ipynb