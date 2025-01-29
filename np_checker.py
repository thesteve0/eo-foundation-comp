import os.path

import numpy as np
from pathlib import Path

## TODO something went wrong during the chipping process. There should always be a 224x244 array
## created for the labels arrays. Doing it this way created error in the data since the labels
## for the broken ones are misaligned with the image. We just take 0s on to the missing dimension at
## the end, but we are not sure where the last array was missing from. Probably had to do something with
## the clip and cell size

print("starting")

BASE_DIR = "/home/spousty/data/remote-sensing-comparison/"

image_shape = (6,224,224)
label_shape = (1,224,224)



image_paths = list(Path(BASE_DIR + "multi-band-image-chips").glob("*.npy"))
label_paths = list(Path(BASE_DIR + "ground-truth-chips").glob("*.npy"))



for image_path in image_paths:
    img = np.load(image_path)
    if not img.shape == image_shape:
        print("Image shape mismatch: " + str(image_path) + " -> " + str(img.shape))
        continue
for label_path in label_paths:
    label = np.load(label_path).astype(np.uint8)
    if not label.shape == label_shape:
        # Grab the h and w of single layer
        current_z, current_h, current_w = label.shape
        # Create new array of target size
        extended_label = np.zeros(label_shape, dtype=label.dtype)

        # Copy existing data
        extended_label[:current_z, :current_h, :current_w] = label

        np.save(label_path, extended_label)


        print("Label shape mismatch: " + str(label_path) + " -> " + str(label.shape))
        continue
print("finished")