import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def box_overlap(base_box, layered_box):
    """
    return what % of base_box's area is covered by layered_box
    boxes are in (x,y,w,h) format, where x,y is the top left of the box
    """
    assert all(base_box) >= 0 and all(layered_box) >= 0
    base_x, base_y, base_w, base_h = base_box
    layered_x, layered_y, layered_w, layered_h = layered_box
    intersection_w = min(base_x + base_w, layered_x + layered_w) - max(0, max(base_x, layered_x))
    intersection_h = min(base_y + base_h, layered_y + layered_h) - max(0, max(base_y, layered_y))
    
    if not (intersection_w and intersection_h): return 0
    return (base_w * base_h) / (intersection_w * intersection_h)

def sample_uniform(a, b):
    return torch.empty(1).uniform_(a, b).item()

def sample_rectangle(w, h, rect_w, rect_h):
    """
    sample a rectangle of side lengths rect_w and rect_h and return in (x,y,w,h) format (unnormalized)
    with x,y being the top left of the box
    """
    square_x = sample_uniform(0, w - rect_w/2.0)
    square_y = sample_uniform(0, h - rect_h/2.0)
    return (square_x, square_y, rect_w, rect_h)

def xywh_to_xyxy(x, y, w, h, xy_center=False):
    """convert [x,y,w,h] -> [x_1, y_1, x_2, y_2]"""
    if xy_center: 
        # (x,y) gives the center of the box
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
    else: 
        # (x,y) gives the upper left corner of the box
        x1, y1 = x, y
        x2, y2 = x+w, y+h
    return int(x1), int(y1), int(x2), int(y2)

def xyxy_to_xywh(x1, y1, x2, y2, xy_center=False):
    """convert [x_1, y_1, x_2, y_2] -> [x,y,w,h]"""
    if xy_center: 
        # (x, y) gives the center of the box
        x, y = (x1+x2)/2, (y1+y2)/2
    else: 
        # (x,y) gives the upper left corner of the box
        x, y = min(x1, x2), min(y1, y2)        
    w = abs(x2-x1)
    h = abs(y2-y1)
    return int(x), int(y), int(w), int(h)

def create_mask_from_bboxes(bboxes, mask_wh: Image.Image):
    """Create an Image mask from a set of bboxes"""
    img_w, img_h = mask_wh
    mask_im = Image.new("L", mask_wh, 0) # make a mask & fill with black       
    for bbox in bboxes:
        x,y,w,h = bbox
        draw = ImageDraw.Draw(mask_im) # make canvas
        xyxy = xywh_to_xyxy(
            x * img_w,  # fraction -> pixels
            y * img_h, 
            w * img_w, 
            h * img_h,
        )
        draw.rectangle(xyxy, fill=255) # draw a rectangle of white where bbox is
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(5))
    return mask_im

def get_surrounding_rectangle(mask: Image.Image):
    """Create an Image mask using the smallest rectangle that surrounds the current mask"""
    mask_wh = mask.size
    y_objects, x_objects = np.where(np.array(mask))
    mask_im = Image.new("L", mask_wh, 0) # make a mask & fill with black
    draw = ImageDraw.Draw(mask_im) # make canvas
    draw.rectangle((
        np.min(x_objects),
        np.min(y_objects), 
        np.max(x_objects), 
        np.max(y_objects),
    ), fill=255)
    return mask_im