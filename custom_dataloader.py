try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskDINO
from maskdino import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskdino_config,
    DetrDatasetMapper,
)
import random
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer
)
import weakref

import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import cv2
import os

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="maskdino")
    return cfg

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def my_dataset_function():
    
    data_img_folder = '/data/GenericCracks/train/inputs_train/1_GenericCrack'
    anno_folder = '/data/GenericCracks/train/json_annos_train/1_GenericCrack'
    all_data_images = sorted(os.listdir(data_img_folder))

    for data_img_filename in all_data_images:
        data_img_path = os.path.join(data_img_folder, data_img_filename)
        anno_filename = data_img_filename + '-annotated.json'
        anno_img_path = os.path.join(anno_folder,anno_filename)
        print(data_img_path, anno_img_path)
        json_format_annotation = read_json(anno_img_path)
    
    return

def load_iap_annotations(json_anno_dir, thing_classes=None, stuff_classes=None, tag_classes=None,
                         mode='sem_seg/instanceboxmask'):
    """

    :param json_anno_dir: location of the IAP json annotation file
    :param stuff_classes: list of stuff classes relates to semantic segmentation labels
    :param thing_classes: list of thing classes relates to instance labels
    :param tag_classes: list of tag classes for whole image classification
    :param mode: mode of task. Can contain multiple modes for one task.
                sem_seg: semantic sementation. stuff_classes must be included
                instance: instance detection.
                    instancebox: only bounding box detection
                    instancemask: instance segmentation
                    instancekey: keypoint detection
    :return: dictionary of the relevant key value pairs based on the mode.
    """
    # initialize not class lists
    if tag_classes is None:
        tag_classes = []
    if stuff_classes is None:
        stuff_classes = []
    if thing_classes is None:
        thing_classes = []

    # read in IAP annotation
    objects = read_json(json_anno_dir)

    # initialize empty annotation sample dictionary
    annotation_sample = {'height': objects['height'], 'width': objects['width']}
    if 'sem_seg' in mode:
        annotation_sample.update({'sem_seg': np.zeros([objects['height'], objects['width']])})
    if 'instance' in mode:
        annotation_sample.update({'instances_cls': []})
        if 'box' in mode:
            annotation_sample.update({'instances_box': []})
        if 'mask' in mode:
            annotation_sample.update({'instances_mask': []})
        if 'key' in mode:
            raise NotImplementedError
    if 'classification' in mode:
        annotation_sample['classification'] = 0

    # go through tags
    if 'tags' in objects and objects['tags'] and 'classification' in mode:
        for tag in objects['tags']:
            if 'name' and 'type' and 'tag' in tag:
                tag_name = tag['name']
                tag_class = tag['tag']
                if tag_class in tag_classes:
                    annotation_sample['classification'] = 1 + tag_classes.index(tag_class)
    # go through labels
    if 'labels' in objects and objects['labels'] and ('sem_seg' in mode or 'instance'):
        for label in objects['labels']:
            if 'name' and 'type' in label:
                class_name = label['name']
                class_type = label['type']

                stuff_class_indexs = [class_name in s for s in stuff_classes] if stuff_classes is not None else None
                thing_class_indexs = [class_name in s for s in thing_classes] if thing_classes is not None else None
                if any(stuff_class_indexs) or any(thing_class_indexs):
                    for annotation in label['annotations']:
                        if 'sem_seg' in mode:
                            if 'rle' in annotation['type']:
                                for x in annotation['segmentation']:
                                    annotation_sample['sem_seg'][x[1], x[0]:x[0] + x[2]] = \
                                        stuff_class_indexs.index(True) + 1
                            if 'polygon' in annotation['type'] or 'boundingBox' in annotation['type'] \
                                    or 'boundingbox' in annotation['type']:
                                annotation_sample['sem_seg'] = fill_mask(annotation_sample['sem_seg'],
                                                                         annotation['segmentation'],
                                                                         value=stuff_class_indexs.index(True) + 1)
                            if 'path' in annotation['type']:
                                raise NotImplementedError
                        if 'instance' in mode and class_type == 'boundingBox' or class_type == 'boundingbox':
                            annotation_sample['instances_cls'].append(
                                thing_class_indexs.index(True))
                            if 'mask' in mode:
                                mask = np.zeros([objects['height'], objects['width']], dtype=np.uint8)
                                mask = fill_mask(mask,
                                                 annotation['segmentation'],
                                                 value=1)
                                annotation_sample['instances_mask'].append(mask)

                                # xs = annotation['segmentation'][:-1:2]
                                # ys = annotation['segmentation'][1::2]
                                #
                                # pts = np.array([[x, y] for x, y in zip(xs, ys)], np.int32)
                                # mask = cv2.fillPoly(mask, [pts], 1)

                            elif 'box' in mode:
                                annotation_sample['instances_box'].append(create_bbox(annotation['segmentation']))
                            elif 'key' in mode:
                                raise NotImplementedError

    return annotation_sample

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--EVAL_FLAG', type=int, default=1)
    args = parser.parse_args()
    cfg = setup(args)
    # print(cfg)
    json_format_annotation = read_json('/data/GenericCracks/train/json_annos_train/1_GenericCrack/555117595.41836__1368.JPG-annotated.json')
    # print(json_format_annotation['labels'].keys())
    iap_anno = load_iap_annotations('/data/GenericCracks/train/json_annos_train/1_GenericCrack/555117595.41836__1368.JPG-annotated.json')
    # my_dataset_function()
    print(iap_anno.keys())

    