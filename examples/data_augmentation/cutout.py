# Adapted from https://github.com/YBZh/Bridging_UDA_SSL

import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from data_augmentation.utils import *

ACCEPTABLE_OVERLAP_THRESHOLD = 0.5
class Cutout:
    def __init__(self, labeled_dataset, bbox_aware=False):
        self.bbox_aware = bbox_aware
        self.dataset = labeled_dataset
        self.nargs = 2

    def __call__(self, img, ix):
        # get bboxes if wanted and available
        if self.bbox_aware:
            try:
                bboxes, conf = self.dataset.get_bbox(ix) 
                if conf < 0.5: bboxes = []
                # unnormalize bboxes
                bboxes = [xywh_to_xyxy(
                    bbox[0] * img.width,
                    bbox[1] * img.height,
                    bbox[2] * img.width,
                    bbox[3] * img.height,
                ) for bbox in bboxes]
            except:
                bboxes = None
        else:
            bboxes = None

        # apply cutout
        cutout_val = sample_uniform(0, 1) * 0.5
        # convert size of box from range (0, 0.5) to (0, 0.5*w)
        assert 0.0 <= cutout_val <= 0.5
        cutout_val = cutout_val * img.size[0]
        img = self.draw_bbox(img, cutout_val, bboxes=bboxes)
        return img

    def draw_bbox(self, img, v, bboxes=None, num_tries=3):  # [0, 60] => percentage: [0, 0.2]
        """
        given v in (0, img.width), sample a square of width v and height v
        if bboxes is not None: attempt to sample squares that do not completely obscure the bboxes
        """
        if v < 0: return img
        w, h = img.size
        
        if bboxes is not None and len(bboxes):
            tries = 1
            xy = sample_rectangle(w, h, v, v)
            while min([box_overlap(bbox, xy) for bbox in bboxes]) > ACCEPTABLE_OVERLAP_THRESHOLD and tries < num_tries:
                xy = sample_rectangle(w, h, v, v)
                tries += 1
        else:
            xy = sample_rectangle(w, h, v, v)
        
        color = (125, 123, 114)
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xywh_to_xyxy(*xy), color)
        return img
