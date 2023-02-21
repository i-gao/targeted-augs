import numpy as np
import torch
from wilds.common.utils import MixedY
from data_augmentation.utils import sample_rectangle

def initialize_batch_transform(
    batch_transform_name, config, dataset, batch_transform_kwargs, transform_p=1.0,
):
    if batch_transform_name is None:
        return None

    assert dataset.is_classification
    assert config.algorithm == 'ERM'

    if batch_transform_name == 'mixup':
        augmentation = MixUp(p=transform_p, **batch_transform_kwargs)
    elif batch_transform_name == 'cutmix':
        augmentation = CutMix(p=transform_p, **batch_transform_kwargs)
    else:
        raise ValueError(f"{batch_transform_name} not recognized")
    return augmentation

#################

class BatchTransform:
    """
    These augmentations operate on batches of examples provided by a torch dataloader, rather than 
    individual examples. Though we initialize a BatchTransform object early on and pass it into train(),
    the transform isn't actually called until train() and we start iterating through the data loader.
    """
    def __init__(self, p):
        self.p = p # probability of applying the augmentation to a batch
    
    def __call__(self, x, y, m):
        assert torch.is_tensor(x) and torch.is_tensor(y) and torch.is_tensor(m)
        assert not isinstance(y, MixedY) # not already mixed
        assert y.ndim == 1  # simple prediction only

        random_apply =  (self.p >= torch.rand(1)).item()
        if (random_apply == True): # with some p, transform the batch
            with torch.no_grad():
                return self._transform_batch(x, y, m)
        else:
            return x, y, m # do nothing
        
    def _transform_batch(self, x, y, m):
        raise NotImplementedError

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for k, v in self.__dict__.items(): 
            format_string += f" {k}={v},"
        format_string += ")"
        return format_string

class MixUp(BatchTransform):
    def __init__(self, p, alpha, within_y=False):
        assert alpha > 0
        self.alpha = alpha
        self.within_y = within_y
        super().__init__(p)

    def _transform_batch(self, x, y, m):
        """
        Assume:
            - x is a tensor of shape (B, ...) where B is the batch size
            - y is a tensor of shape (B, )
            - m is a tensor of shape (B, ...)
        """
        batch_size = x.shape[0]
        lmbda = np.random.beta(self.alpha, self.alpha)

        if self.within_y:
            # for each x, select another x' that has the same y, and include the option of selecting x itself
            x1, x2 = x, x[[np.random.choice(np.where(y == yp)[0]) for yp in y]]
            y1, y2 = y, y
        else:
            rand_index = torch.randperm(batch_size) # on cpu
            y1, y2 = y, y[rand_index]
            x1, x2 = x, x[rand_index]

        x = lmbda * x1 + (1-lmbda) * x2
        y = MixedY(y1, y2, lmbda)
        m = m # don't mix metadata -- shouldn't affect ERM
        return x, y, m

class CutMix(BatchTransform):
    def __init__(self, p, alpha, within_y=False):
        assert alpha > 0
        self.alpha = alpha
        self.within_y = within_y
        super().__init__(p)

    def _transform_batch(self, x, y, m):
        """
        Assume:
            - x is a tensor of shape (B, ...) where B is the batch size
            - y is a tensor of shape (B, )
            - m is a tensor of shape (B, ...)
        """
        batch_size, _, w, h = x.shape
        lmbda = np.random.beta(self.alpha, self.alpha)
        
        bbox = sample_rectangle(w, h, w*np.sqrt(1-lmbda), h*np.sqrt(1-lmbda))
        bbox_l, bbox_t = int(bbox[0]), int(bbox[2])
        bbox_r, bbox_b = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        if self.within_y:
            # for each x, select another x' that has the same y, and include the option of selecting x itself
            rand_index = [np.random.choice(np.where(y == yp)[0]) for yp in y]
        else:
            rand_index = torch.randperm(batch_size) # on cpu
        
        x[:, :, bbox_l:bbox_r, bbox_t:bbox_b] = x[rand_index, :, bbox_l:bbox_r, bbox_t:bbox_b] 
        lmbda = 1 - ((bbox_r - bbox_l) * (bbox_b - bbox_t)) / (w * h) # adjust to be the pixel ratio
        y1, y2 = y, y[rand_index]
        y = MixedY(y1, y2, lmbda)            
        m = m # don't mix metadata -- shouldn't affect ERM
        return x, y, m