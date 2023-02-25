import torch
import numpy as np
from PIL import Image

from utils import isin
from data_augmentation.utils import create_mask_from_bboxes
from data_augmentation.utils import *

class CopyPasteAugment:
    def __init__(self, labeled_dataset, same_cluster=False, same_y_observed=False, translate_upper=None, translate_lower=None, resize_upper=None, resize_lower=None): 
        """
        Assumes that the dataset has a get_bbox method that returns the bounding box of the object in the image.
        If a get_mask method is also present, it will be used instead of the bounding box.

        Assumes that the dataset has a empty_indices attribute that is a dict of the form {'train': [list of indices], ...},
        where the indices are the indices of empty (no object) examples in the dataset.

        Initialize the augmentation.
            - same_cluster samples an empty image from the same cluster of cameras as the incoming input
            - same_y_observed samples an empty image from a camera that in the dataset observes an instance of the same y as the incoming input
        """
        assert labeled_dataset.dataset_name in ("iwildcam", "birdcalls") # must have bboxes / masks
        assert hasattr(labeled_dataset, 'get_bbox'), "You might be using a dataset class from the default wilds package. This code assumes you're using our modified version that returns bboxes, masks, etc."
        assert not (translate_upper is None) ^ (translate_lower is None) 
        assert not (resize_upper is None) ^ (resize_lower is None) 
        assert np.sum([same_cluster, same_y_observed]) <= 1, "at most one of same_* flags can be on at a time"

        self.translate_range = (translate_lower, translate_upper) if translate_upper is not None else None
        self.resize_range = (resize_lower, resize_upper) if resize_upper is not None else None
        self.dataset = labeled_dataset

        self.nargs = 2 # for the Transform class, to signal we want both img and ix passed to __call__

        self.classes_to_not_augment = [0] # 0 is the empty class for iwildcam, birdcalls

        # prepare a list of empty images to paste onto
        if 'train' in self.dataset.empty_indices:
            self.empty_indices = self.dataset.empty_indices['train']
        else:
            self.empty_indices = np.array([], dtype=int)

        # save other kwargs
        if same_cluster: assert hasattr(self.dataset, 'cluster_array')
        self.same_cluster = same_cluster
        self.same_y_observed = same_y_observed
    
    def __call__(self, img, ix):
        """
        Cut & paste the object in the img onto an empty image. Requires both the PIL image and the index of the example in the dataset.
        """
        dataset = self.dataset

        # don't transform empty images
        y = dataset.y_array[ix].item()
        if y in self.classes_to_not_augment: return img # IDENTITY CASE

        # get relevant annotations
        if hasattr(dataset, 'get_mask'): mask = dataset.get_mask(ix, resize_wh=img.size) # this returns an image
        else: mask = None
        if mask is None or np.all(np.array(mask) == 0): # there exist cases where a mask and bboxes exist, but the mask is empty
            bboxes, conf = dataset.get_bbox(ix)
            if conf < 0.5 or len(bboxes) == 0: return img # IDENTITY CASE
            mask = create_mask_from_bboxes(bboxes, img.size)
            assert not np.all(np.array(mask) == 0), "Mask shouldn't be empty by this point."

        # set randomly sampled translation / resizing hparams
        if self.translate_range is not None:
            translate_xy = np.random.uniform(low=self.translate_range[0], high=self.translate_range[1], size=2)
            # don't move the center of the bbox off the screen
            x_center, y_center = get_object_center(mask)
            if x_center is not None and y_center is not None: # should be a no-op by this point; masks shouldn't be empty, but maybe the get_object_center fn fails
                x_center, y_center = x_center / img.width, y_center / img.height
                translate_xy[0], translate_xy[1] = np.clip(translate_xy[0], -x_center, 1-x_center), np.clip(translate_xy[1], -y_center, 1-y_center)
        else:
            translate_xy = (0, 0)
        if self.resize_range is not None:
            resize_ratio = np.random.uniform(low=self.resize_range[0], high=self.resize_range[1], size=1)
        else:
            resize_ratio = None

        # sample the empty background
        if self.same_cluster: 
            bg_img = self.get_empty_img_from_cluster(dataset.cluster_array[ix])
            if bg_img is None: return img # IDENTITY CASE
        elif self.same_y_observed: 
            bg_img = self.get_empty_img_by_y(y=dataset.y_array[ix].item())
            if bg_img is None: return img # IDENTITY CASE
        else: 
            bg_img = self.get_empty_img()
        
        bg_img = bg_img.copy().resize(img.size)

        # resize if necessary
        if resize_ratio is not None:
            img, mask = resize_object(img, mask, resize_ratio)
        
        # paste
        bg_img.paste(
            img,
            box=(int(translate_xy[0] * bg_img.width), int(translate_xy[1] * bg_img.height)),
            mask=mask
        )
        return bg_img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for k in ('translate_range', 'resize_range'):
            format_string += f" {k}={getattr(self, k)},"
        for k in ('same_cluster', 'same_y_observed'):
            if getattr(self, k): format_string += f" {k}={getattr(self, k)}"
        format_string += ")"
        return format_string

    ######################################

    def get_empty_img(self):
        """pick an empty image from any camera location"""  
        return self._sample_empty_img_given_mask(
            torch.ones(len(self.empty_indices), dtype=bool)
        )
    
    def get_empty_img_from_cluster(self, cluster):
        """pick an empty image from a camera cluster; some clusters have no empty images"""
        labeled_mask = (self.dataset.cluster_array[self.empty_indices] == cluster)
        return self._sample_empty_img_given_mask(labeled_mask)

    def get_empty_img_by_y(self, y):
        """pick an empty image from a location that contains an example with label y"""
        labeled_mask = isin(self.dataset.location_array, self.dataset.y_to_observed_locs[y])[self.empty_indices]
        return self._sample_empty_img_given_mask(labeled_mask)

    ################################

    def _sample_empty_img_given_mask(self, labeled_mask):
        """return a random item from empty_indices that satisfies the given boolean masks"""
        assert len(labeled_mask) == len(self.empty_indices)
        # empty indices are numpy arrays, so make masks numpy

        labeled_mask = labeled_mask.numpy()
        if np.all(labeled_mask == 0):
            return None

        empty_ix = np.random.choice(self.empty_indices[labeled_mask])
        return self.dataset.get_input(empty_ix)

############

def get_object_center(mask: Image):
    """return a point (x,y) representing the 'center' of the objects"""
    y_objects, x_objects = np.where(np.array(mask))
    if not (len(y_objects) or len(x_objects)): 
        return None, None
    else:
        return int(np.median(x_objects)), int(np.median(y_objects))

def resize_object(img: Image, mask, resize_ratio: float):
    """return mask and Image resized to the resize_ratio"""
    w, h = img.size
    new_w, new_h = (np.array(img.size) * resize_ratio).astype(int)

    x_object, y_object = get_object_center(mask)
    if y_object is None or x_object is None: return img, mask
    
    l, t, r, b = xywh_to_xyxy(x_object, y_object, w, h, xy_center=True)
    img = img.crop((l, t, r, b)).resize((new_w, new_h)).transform(
        (w, h), 
        method=Image.AFFINE,
        data=(1, 0, (new_w/2 - x_object), 0, 1, (new_h/2 - y_object))
    ) # center object, crop, and then move back
    mask = mask.crop((l, t, r, b)).resize((new_w, new_h)).transform(
        (w, h), 
        method=Image.AFFINE,
        data=(1, 0, (new_w/2 - x_object), 0, 1, (new_h/2 - y_object))
    )
    return img, mask