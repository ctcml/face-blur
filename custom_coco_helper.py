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


# Mock up Instance class used to create Instances with just one detected object
class MyInstance:
    def __init__(self, pred_classes=None, pred_boxes=None, pred_masks=None, scores=None):
        self.pred_classes = pred_classes
        self.pred_boxes = pred_boxes
        self.pred_masks = pred_masks
        self.scores = scores
        
    def has(self, s):
        if s == 'pred_classes':
            return self.pred_classes is not None
        if s == 'pred_boxes':
            return self.pred_boxes is not None
        if s == 'pred_masks':
            return self.pred_masks is not None
        if s == 'scores':
            return self.scores is not None        
        

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
            im = cv2.imread(self.img_paths[0])
        except Exception as e:
            print(e)
    return im

    def inference(self, im):
        outputs = self.predictor(im)
        return outputs
        
        
        
def visualize(im, instances):
    # Visualize the image with object masks
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instances)
    plt.figure(figsize = (200,20))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))