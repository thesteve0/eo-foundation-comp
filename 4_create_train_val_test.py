import os
from pathlib import Path

import fiftyone as fo
import tempfile

TRAIN_CHIP_DIR = "/home/spousty/git/clay/model/data/train/chips"
TRAIN_LABEL_DIR = "/home/spousty/git/clay/model/data/train/labels"
VALID_CHIP_DIR =  "/home/spousty/git/clay/model/data/valid/chips"
VALID_LABEL_DIR =  "/home/spousty/git/clay/model/data/valid/labels"
TEST_CHIP_DIR =  "/home/spousty/git/clay/model/data/test/chips"
TEST_LABEL_DIR =  "/home/spousty/git/clay/model/data/test/labels"

OUTPUT_LOCATIONS = dict(
    train = {"chip": TRAIN_CHIP_DIR, "label": TRAIN_LABEL_DIR},
    valid = {"chip": VALID_CHIP_DIR, "label": VALID_LABEL_DIR},
    test = {"chip": TEST_CHIP_DIR, "label": TEST_LABEL_DIR},
)

for set_names, values in OUTPUT_LOCATIONS.items():
    for value in values:
        Path(values[value]).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    '''
    Our dataset has about 7583 samples
    3000 for fine tuning = 
    2500 training
    500 for validation
    ---------------
    500 for testing
    '''

    print("starting")

    dataset = fo.load_dataset("rs_chip_images")
    if fo.dataset_exists("clay_data"):
        fo.delete_dataset("clay_data")

    clay_dataset = dataset.shuffle(seed=0.5).limit(3500).clone("clay_data", persistent=True)
    clay_dataset.take(500).tag_samples("test")
    clay_dataset.match_tags("test", bool=False).take(2500).tag_samples("train")
    clay_dataset.match_tags(["test", "train"], bool=False).tag_samples("valid")

    train = clay_dataset.match_tags("train")
    test = clay_dataset.match_tags("test")
    valid = clay_dataset.match_tags("valid")

    sets_to_create = dict(train=train, test=test, valid=valid)

    for setname, setview in sets_to_create.items():
        # https://docs.python.org/3/library/os.html#os.symlink
        for sample in setview:
           chip_file = sample["chip_path"]
           chip_file_name = Path(chip_file).name

           label_file = chip_file.replace("_chip", "_lulc_chip").replace("multi-band-image-chips", "ground-truth-chips")
           label_file_name = Path(label_file).name

           chip_link_path = OUTPUT_LOCATIONS[setname]["chip"]
           label_link_path = OUTPUT_LOCATIONS[setname]["label"]
           try:
               os.symlink(chip_file, os.path.join(chip_link_path, chip_file_name))
           except:
               pass
           try:
               os.symlink(label_file, os.path.join(label_link_path, label_file_name))
           except:
               pass

    print("finished")