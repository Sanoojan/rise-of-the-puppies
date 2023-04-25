"""
# TO RUN:
python -W ignore Detectron2-modif/tools/lazyconfig_train_net.py \
    --config-file tasks/vitdetB_run_config.py

# TO EVALUATE
python -W ignore Detectron2-modif/tools/lazyconfig_train_net.py \
    --config-file configs/path/to/config.py \
        --eval-only train.init_checkpoint=/path/to/model_checkpoint

"""


from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate
from detectron2.data.samplers import RepeatFactorTrainingSampler

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L

from detectron2.data.datasets import register_coco_instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import (COCOEvaluator, DatasetEvaluators)


##------------------------------------------------------------------------------


OUTPUT_DIRECTORY="OUTPUT/#DUMMY/Ex20-ViTDet_attempt/"

WEIGHT_TO_LOAD="/home/joseph.benjamin/LABS/cv703/project-work/Model-Backups/ViTDetB-model_final_61ccd1.pkl"
RESUME= False

iSAID_DATASET_PATH="/apps/local/shared/CV703/datasets/iSAID/iSAID_patches/"


##------------------------------------------------------------------------------

def register_dataset(path):
    register_coco_instances("iSAID_train", {},
                            f"{path}/train/instancesonly_filtered_train.json",
                            f"{path}/train/images/")
    register_coco_instances("iSAID_val", {},
                            f"{path}/val/instancesonly_filtered_val.json",
                            f"{path}/val/images/")
    register_coco_instances("iSAID_test", {},
                            f"{path}/test/test_info.json",
                            f"{path}/test/images/")

register_dataset(iSAID_DATASET_PATH)


# Data using LSJ
image_size = 800
dataloader = model_zoo.get_config("common/data/coco.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=image_size, target_width=image_size
    ),
    L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.total_batch_size = 2
# recompute boxes due to cropping
dataloader.train.mapper.recompute_boxes = True

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
]


dataloader.train.dataset.names = "iSAID_train"
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
        dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
    )
)
dataloader.test.dataset.names = "iSAID_val"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    max_dets_per_image=300,
)



model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.roi_heads.num_classes = 15
model.roi_heads.box_predictor.test_topk_per_image=300

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = (
    # "detectron2://ImageNetPretrained/MAE/mae_pretrain_vit_base.pth?matching_heuristics=True"
    WEIGHT_TO_LOAD
)
train.output_dir = OUTPUT_DIRECTORY

# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 90000
train.resume=True


lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        # milestones=[163889, 177546],
        milestones=[60000, 80000],
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
