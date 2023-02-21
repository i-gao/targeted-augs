import torch
from PIL import Image
from data_augmentation.utils import *

ACCEPTABLE_OVERLAP_THRESHOLD = 0.2
class RandomCrop:
    def __init__(self, size, labeled_dataset, num_tries=3, bbox_aware=False):
        assert size[0] > 0 and size[1] > 0
        self.width, self.height = size
        self.bbox_aware = bbox_aware
        self.num_tries = num_tries
        self.dataset = labeled_dataset

        self.nargs = 2

    def __call__(self, img, ix):
        w, h = img.size
        # get bboxes if wanted and available
        if self.bbox_aware:
            try:
                bboxes, conf = self.dataset.get_bbox(ix)
                if conf < 0.5: bboxes = []
                # unnormalize bboxes
                bboxes = [xywh_to_xyxy(
                    bbox[0] * w,
                    bbox[1] * h,
                    bbox[2] * w,
                    bbox[3] * h,
                ) for bbox in bboxes]
            except:
                bboxes = None
        else:
            bboxes = None

        # set crop area
        if bboxes is not None and len(bboxes):
            tries = 1
            xy = sample_rectangle(w, h, self.width, self.height)
            while max([box_overlap(bbox, xy) for bbox in bboxes]) < ACCEPTABLE_OVERLAP_THRESHOLD and tries < self.num_tries:
                xy = sample_rectangle(w, h, self.width, self.height)
                tries += 1
        else:
            xy = sample_rectangle(w, h, self.width, self.height)
        
        # crop & return
        img = img.copy().crop(box=xywh_to_xyxy(*xy))
        return img