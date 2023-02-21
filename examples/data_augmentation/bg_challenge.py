import numpy as np
from PIL import Image


from data_augmentation.utils import get_surrounding_rectangle, create_mask_from_bboxes

class BGChallenge:
    def __init__(self, labeled_dataset, mode="no-fg"): 
        """
        Implementation of some of the augmentations in https://arxiv.org/pdf/2006.09994.pdf.

        Assumes that the dataset has a get_bbox method that returns the bounding box of the object in the image.
        If a get_mask method is also present, it will be used instead of the bounding box.
        """
        assert hasattr(labeled_dataset, 'get_bbox'), "You might be using a dataset class from the default wilds package. This code assumes you're using our modified version that returns bboxes, masks, etc."
        assert mode in ('only-bg-b', 'no-fg', 'only-fg')
        self.dataset = labeled_dataset
        self.mode = mode
        self.nargs = 2

    def __call__(self, img, ix):
        dataset = self.dataset
        
        # get relevant annotations
        if hasattr(dataset, 'get_mask'): mask = dataset.get_mask(ix, resize_wh=img.size) # this returns an image
        else: mask = None
        if mask is None or np.all(np.array(mask) == 0): # there exist cases where a mask and bboxes exist, but the mask is empty
            bboxes, conf = dataset.get_bbox(ix)
            if conf < 0.5 or len(bboxes) == 0: return img # IDENTITY CASE
            mask = create_mask_from_bboxes(bboxes, img.size)
            assert not np.all(np.array(mask) == 0), "Mask shouldn't be empty by this point."

        if self.mode == 'only-bg-b': 
            mask = get_surrounding_rectangle(mask)
        
        # get the background
        if self.mode == 'only-fg': 
            bg_img = Image.new(mode='RGB', size=img.size) # all black 
        else: 
            bg_img = img.copy()

        # get the foreground
        if self.mode != 'only-fg':
            img = Image.new(mode='RGB', size=img.size) # all black 

        # paste
        bg_img.paste(
            img,
            mask=mask
        )
        img.close()
        mask.close()
        return bg_img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"({self.mode})"
        return format_string