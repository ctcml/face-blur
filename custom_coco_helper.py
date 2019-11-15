# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import copy
import time
import os
import datetime
import json

from pycococreatortools import pycococreatortools 


# Mock up Instance class used to create Instances with just one detected object
class MyInstance:
    def __init__(self, img_id, seg_id, img_size, img_path, pred_classes=None, pred_boxes=None, pred_masks=None, scores=None):
        self.img_id = img_id
        self.seg_id = seg_id
        self.img_size = img_size
        self.img_path = img_path
        self.pred_classes = pred_classes
        self.pred_boxes = pred_boxes
        self.pred_masks = pred_masks
        self.scores = scores
        
    # Check whether the specified inference result exists in the instance
    def has(self, s):
        if s == 'pred_classes':
            return self.pred_classes is not None
        if s == 'pred_boxes':
            return self.pred_boxes is not None
        if s == 'pred_masks':
            return self.pred_masks is not None
        if s == 'scores':
            return self.scores is not None
        
    # Return annotation in coco format
    def coco_annotation(self, new_class=None):
        if not new_class:
            true_class = self.pred_classes[0].tolist()
        else:
            true_class = new_clas
        category_info = {'id': true_class, 'is_crowd': 0}
        
        annotation_info = pycococreatortools.create_annotation_info(
            self.seg_id,
            self.img_id,
            category_info,
            self.pred_masks[0],
            tolerance=2,
            bounding_box=self.pred_boxes[0]
        )
        
        return annotation_info
    
    # Return coco format img info
    def coco_img_info(self):
        img_info = pycococreatortools.create_image_info(
            self.img_id,
            self.img_path,
            self.img_size
        )
        
        return img_info
        
    # Create a list of MyInstances from the Detectron Instances
    @classmethod
    def create_instances(cls, img_id, next_seg_id, img_paths, instances, keep_cats=[]):
        # loop through every detected objects in the result and display them one by one
        my_instances = []
        for i in range(len(outputs["instances"].pred_classes)):
            # Only keep specified categories
            pred_classes = outputs["instances"].pred_classes[i:i+1].to("cpu")
            if pred_classes[0] not in keep_ids:
                continue

            # img meta
            img_size = [outputs["instances"].image_size[1], outputs["instances"].image_size[0]]
            img_path = img_paths[i]
                
            my_instance = MyInstance(
                img_id = img_id,
                seg_id = start_seg_id,
                img_size = img_size,
                img_path = img_path,
                pred_classes = pred_classes,
                pred_boxes = outputs["instances"].pred_boxes[i:i+1].to("cpu"),
                pred_masks = outputs["instances"].pred_masks[i:i+1].to("cpu"),
                scores = outputs["instances"].scores[i:i+1].to("cpu"),
            )
            
            my_instances.append(my_instance)
            next_seg_id += 1
        return next_seg_id, my_instances

# Wrapper for running inference
class MyPredictor:
    # Constants
    DATA_DIR = 'data'
    DOWNLOAD_DIR = 'raw'
    IMAGE_PATH = os.path.join(os.getcwd(), DATA_DIR, DOWNLOAD_DIR)

    def __init__(self):
        # Using cascade mask rcnn as the predictor
        self.cfg = get_cfg()
        self.cfg.merge_from_file('./detectron_configs/Cityscapes/cascade_mask_rcnn_R_50_FPN_1x.yaml')
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detectron2/Misc/cascade_mask_rcnn_R_50_FPN_1x/138602847/model_final_e9d89b.pkl"
        self.predictor = DefaultPredictor(cfg)

        self.img_paths = [os.path.join(IMAGE_PATH, img_name) for img_name in os.listdir(IMAGE_PATH)]
        
    def read_im(self, img_id):
        # Images might be corrupted during downloads, or formats might be png
        # Does not happen very often, just skip
        try:
            im = cv2.imread(self.img_paths[img_id])
        except Exception as e:
            print(e)
        return im

    def inference(self, im):
        outputs = self.predictor(im)
        return outputs
        

class CustomCOCOFormatter:
    info = {
        "description": "",
        "url": "",
        "version": "",
        "year": datetime.datetime.now().year,
        "contributor": "",
        "date_created": datetime.datetime.now().isoformat(' ')
    }
    
    licenses = [{
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }]
    
    def __init__(self, categories, annotations, img_info, info={}):
        for k, v in info.items():
            self.info[k] = v
            
        self.categories = categories
        self.annotations = annotations
        self.img_info = img_info
    
    def export(self, file_path):
        to_write = {
            "info": self.info,
            "licenses": self.licenses,
            "categories": self.categories,
            "images": self.img_info,
            "annotations": self.annotations
        }
        
        with open(file_path, 'w') as f:
            json.dump(to_write, f)
    
        
def visualize(im, instances):
    # Visualize the image with object masks
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instances)
    plt.figure(figsize = (200,20))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))