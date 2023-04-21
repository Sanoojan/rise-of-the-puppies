import torch, torchvision

# Import some common libraries
import numpy as np
import sys, os, json, cv2
import time, random

# Import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluators)

# import detectron2.modeling.backbone.acmix_resnet

##======================= Setup ================================================

OUTPUT_DIRECTORY="OUTPUT/ACMixR50FPN_coco/"  # The output directory to save logs and checkpoints

RESUME = False
CONFIG_FILE_PATH="COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"  # The detectron2 config file for R50-FPN Faster-RCNN


iSAID_DATASET_PATH="/nfs/projects/cv703/jazz-cvgroup-9/iSAID_patches"  # Path to iSAID dataset

if not os.path.exists(OUTPUT_DIRECTORY):
    os.makedirs(OUTPUT_DIRECTORY)
elif RESUME:
    pass
else:
    raise Exception(f"Directory Exists ... !!! {OUTPUT_DIRECTORY}")

setup_logger(f"{OUTPUT_DIRECTORY}/log.txt")
print(OUTPUT_DIRECTORY)

##-------------------- Data Set ------------------------------------------------
def register_dataset(path):
    register_coco_instances("iSAID_train", {},
                            f"{path}/train/instancesonly_filtered_train.json",
                            f"{path}/train/images/")
    register_coco_instances("iSAID_val", {},
                            f"{path}/val/instancesonly_filtered_val.json",
                            f"{path}/val/images/")

register_dataset(iSAID_DATASET_PATH)

##-------------------- Config ------------------------------------------------


def prepare_config(config_path, **kwargs):
    # Parse the expected key-word arguments
    output_path = kwargs["output_dir"]

    # Create and initialize the config
    cfg = get_cfg()
    cfg.SEED = 26911042  # Fix the random seed to improve consistency across different runs
    cfg.OUTPUT_DIR = output_path
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.DATASETS.TRAIN = ("iSAID_train",)
    cfg.DATASETS.TEST = ("iSAID_val",)
    cfg.DATALOADER.NUM_WORKERS = 16

    cfg.MODEL.BACKBONE.NAME = "build_acmix_resnet_fpn_backbone"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 15
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    # Training schedule - equivalent to 0.25x schedule as per Detectron2 criteria
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.STEPS = (60000, 80000)
    cfg.SOLVER.MAX_ITER = 90000

    return cfg

d2_config = prepare_config(CONFIG_FILE_PATH, output_dir=OUTPUT_DIRECTORY)

##======================= Training Routine =====================================

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return DatasetEvaluators([COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)])

trainer = Trainer(d2_config)  # Create Trainer
trainer.resume_or_load(resume=RESUME)  # Set resume=True if intended to automatically resume training

start_time = time.time()
trainer.train()  # Train and evaluate the model
print("Train TIMING", time.time() - start_time, ":"*50)

# ## Evaluate the trained model on validation set

# Path to the trained ".pth" model file
MODEL_CHECKPOINTS_PATH=f"{OUTPUT_DIRECTORY}/model_final.pth"
d2_config.MODEL.WEIGHTS = MODEL_CHECKPOINTS_PATH  # Update the weights path in the config
d2_config.OUTPUT_DIR = f"{OUTPUT_DIRECTORY}/validate"  # Update the output directory path

setup_logger(f"{d2_config.OUTPUT_DIR}/log.txt")  # Setup the logger
model = Trainer.build_model(d2_config)  # Build model using Trainer
DetectionCheckpointer(model, save_dir=d2_config.OUTPUT_DIR).load(MODEL_CHECKPOINTS_PATH)  # Load the checkpoints

start_time = time.time()
Trainer.test(d2_config, model)  # Test the model on the validation set
print("Valid TIMING", time.time() - start_time, ":"*50)


##======================= Test Submit ==========================================

### Register Test Dataset
def register_test_datase(path):
    register_coco_instances("iSAID_test", {},
                            f"{path}/test/test_info.json",
                            f"{path}/test/images/")

register_test_datase(iSAID_DATASET_PATH)

### Update config
# Path to the trained ".pth" model file
MODEL_CHECKPOINTS_PATH=f"{OUTPUT_DIRECTORY}/model_final.pth"
d2_config.MODEL.WEIGHTS = MODEL_CHECKPOINTS_PATH  # Update the weights path in the config
d2_config.DATASETS.TEST = ("iSAID_test",)  # Set the test set as "iSAID_test"
d2_config.OUTPUT_DIR = f"{OUTPUT_DIRECTORY}/test"  # Update the output directory path

setup_logger(f"{d2_config.OUTPUT_DIR}/log.txt")  # Setup the logger
model = Trainer.build_model(d2_config)  # Build model using Trainer
DetectionCheckpointer(model, save_dir=d2_config.OUTPUT_DIR).load(MODEL_CHECKPOINTS_PATH)  # Load the checkpoints

start_time = time.time()
Trainer.test(d2_config, model)  # Test the model on the Test set
print("TEST TIMING", time.time() - start_time, ":"*50)
