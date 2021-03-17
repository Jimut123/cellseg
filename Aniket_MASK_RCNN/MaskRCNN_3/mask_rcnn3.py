import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
from imgaug import augmenters as iaa
# %tensorflow_version 1.x
import tensorflow.compat.v1 as tf
# Root directory of the project
ROOT_DIR = os.path.abspath("./Mask_RCNN/")
sys.path.append(ROOT_DIR)
# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # find local version
import coco

# %matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class BloodConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "blood"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # Background + basophil et al

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 10

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between basophil and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet101"

    # # Input image resizing
    # # Random crops of size 64x64
    # IMAGE_RESIZE_MODE = "crop"
    # IMAGE_MIN_DIM = 64
    # IMAGE_MAX_DIM = 64
    # IMAGE_MIN_SCALE = 2.0

    # # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    # POST_NMS_ROIS_TRAINING = 1000
    # POST_NMS_ROIS_INFERENCE = 2000

    # # Non-max suppression threshold to filter RPN proposals.
    # # You can increase this during training to generate more propsals.
    # RPN_NMS_THRESHOLD = 0.9

    # # How many anchors per image to use for RPN training
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # # Image mean (RGB)
    # MEAN_PIXEL = np.array([43.53, 39.56, 48.22])


    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class BloodInferenceConfig(BloodConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    # IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

name_dict = {
        "BA": 1,
        "EO": 2,
        "ERB": 3,
        "IG": 4,
        "MMY": 4,
        "MY": 4,
        "PMY": 4,
        "LY": 5,
        "MO": 6,
        "BNE": 7,
        "NEUTROPHIL": 7,
        "SNE": 7,
        "PLATELET": 8,
        
}

class_names = [
      "basophil","eosinophil", "erythroblast", "ig", "lymphocyte", "monocyte", "neutrophil", "platelet"
]

class BloodDataset(utils.Dataset):

    def load_blood(self, dataset_dir, subset):
        """Load a subset of the nuclei dataset.
        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("blood", 1, "basophil")
        self.add_class("blood", 2, "eosinophil")
        self.add_class("blood", 3, "erythroblast")
        self.add_class("blood", 4, "ig")
        self.add_class("blood", 5, "lymphocyte")
        self.add_class("blood", 6, "monocyte")
        self.add_class("blood", 7, "neutrophil")
        self.add_class("blood", 8, "platelet")
        
        # Which subset?

        assert subset in ["train", "val"]

        for class_name in class_names:
          dataset_class_dir = os.path.join(dataset_dir, class_name, subset)
          image_dir = os.path.join(dataset_class_dir, "images")
          print(image_dir)
          # Get image ids from directory names
          image_ids = list(map(lambda x: x.split('.')[0], next(os.walk(image_dir))[2]))
      
          # Add images
          for image_id in image_ids:
            image_prefix = image_id.split("_")[0]
            num_ids = [ name_dict[image_prefix] ]
            self.add_image(
                "blood",
                image_id=image_id,
                num_ids=num_ids,
                path=os.path.join(dataset_class_dir, "images/{}.jpg".format(image_id)))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .png image
        mask = []
        mask_file_name = os.path.join(mask_dir, "mask_{}.png".format(info["id"]))
        m = skimage.io.imread(mask_file_name).astype(np.bool)
        mask.append(m)
        num_ids = np.array(info['num_ids'], dtype=np.int32)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, num_ids # np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "blood":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, dataset_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = BloodDataset()
    dataset_train.load_blood(dataset_dir, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BloodDataset()
    dataset_val.load_blood(dataset_dir, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])


    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    history_network_heads = model.train(dataset_train, dataset_val,
                                learning_rate=config.LEARNING_RATE,
                                epochs=10,
                                augmentation=augmentation,
                                layers='heads')
    
    hist_df_nh = pd.DataFrame(history_network_heads.history)
    hist_json_file = 'history_network_heads.json'
    with open(hist_json_file, mode='w') as f:
        hist_df_nh.to_json(f)

    print("Train all layers")
    history_dataset_train = model.train(dataset_train, dataset_val,
                                learning_rate=config.LEARNING_RATE,
                                epochs=15,
                                augmentation=augmentation,
                                layers='all')
    
    hist_df_dt = pd.DataFrame(history_dataset_train.history)
    hist_json_file = 'history_dataset_train.json'
    with open(hist_json_file, mode='w') as f:
        hist_df_dt.to_json(f)
    

config = BloodConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=MODEL_DIR)
# Start from ImageNet trained weights
weights_path = model.get_imagenet_weights()
model.load_weights(weights_path, by_name=True)

train(model, "/content/dataset")


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

inference_config = BloodInferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=inference_config,
                                  model_dir=MODEL_DIR)
weights_path = model.find_last()
model.load_weights(weights_path, by_name=True)

def searchKeysByVal(dict, byVal):
    keysList = []
    itemsList = dict.items()
    for item in itemsList:
        if item[1] == byVal:
            keysList.append(item[0])
    return keysList


# Test on a random image
dataset_val = BloodDataset()
dataset_val.load_blood("/content/dataset", "val")
dataset_val.prepare()
image_id = random.choice(dataset_val.image_ids)

original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_val, inference_config, 
                          image_id, use_mini_mask=False)


log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

results = model.detect([original_image], verbose=1)

r = results[0]
print("True: ", dataset_val.image_reference(image_id).split("_")[0])
print("Prediction")
for c, s in zip(r['class_ids'], r['scores']):
  print(searchKeysByVal(name_dict, c), " ===> ", s)
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())
