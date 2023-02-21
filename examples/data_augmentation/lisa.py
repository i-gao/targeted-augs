import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from data_augmentation.utils import sample_rectangle

class LISAMixUp:
    """
    Unlike MixUp within-y, which mixes data with other points in a batch, LISA samples NEW data from the dataset to mix in 
    with the batch, meaning it may use twice the number of data points as normal MixUp.
    """
    def __init__(self, dataset, grouper, alpha, split="train"):
        assert alpha > 0
        self.alpha = alpha
        self.dataset = dataset
        self.grouper = grouper
        self.split = split
        self.nargs = 2

        # save a lookup of each group to all applicable indices in dataset
        self.dataset_g = self.grouper.metadata_to_group(self.dataset.metadata_array)

    def __call__(self, img, ix):
        x = to_tensor(img)
        lmbda = np.random.beta(self.alpha, self.alpha)

        # try to sample another example with the same y but different domain
        y, m = self.dataset.y_array[ix], self.dataset.metadata_array[ix, :]
        g = self.grouper.metadata_to_group(torch.unsqueeze(m, dim=0))
        options = np.where((self.dataset_g != g) & (self.dataset.y_array == y) & (self.dataset.split_array == self.dataset.split_dict[self.split]))[0]
        
        # if such an x' does not exist, don't mix the example
        if len(options) == 0:
            return img

        # resize x' to match our image, mix, and then return as PIL
        # NOTE: only x is mixed; metadata is not mixed -- shouldn't affect erm
        x2, _, _ = self.dataset[np.random.choice(options)]
        x2 = to_tensor(x2.resize(img.size))
        return to_pil_image(lmbda * x + (1-lmbda) * x2)

class LISACutMix:
    """
    Unlike CutMix within-y, which mixes data with other points in a batch, LISA samples NEW data from the dataset to mix in 
    with the batch, meaning it may use twice the number of data points as normal CutMix.
    Also unlike CutMix, we blend the box regions smoothly according to some lambda.
    """
    def __init__(self, dataset, grouper, alpha, split="train"):
        assert alpha > 0
        self.alpha = alpha
        self.dataset = dataset
        self.grouper = grouper
        self.split = split
        self.nargs = 2

        # save a lookup of each group to all applicable indices in dataset
        self.dataset_g = self.grouper.metadata_to_group(self.dataset.metadata_array)

    def __call__(self, img, ix):
        x = to_tensor(img)
        lmbda = np.random.beta(self.alpha, self.alpha)

        # try to sample another example with the same y but different domain
        y, m = self.dataset.y_array[ix], self.dataset.metadata_array[ix, :]
        g = self.grouper.metadata_to_group(torch.unsqueeze(m, dim=0))
        options = np.where((self.dataset_g != g) & (self.dataset.y_array == y) & (self.dataset.split_array == self.dataset.split_dict[self.split]))[0]
        
        # if such an x' does not exist, don't mix the example
        if len(options) == 0:
            return img

        # resize x' to match our image, mix, and then return as PIL
        # NOTE: only x is mixed; metadata is not mixed -- shouldn't affect erm
        x2, _, _ = self.dataset[np.random.choice(options)]
        x2 = to_tensor(x2.resize(img.size))

        w, h = img.size
        bbox = sample_rectangle(w, h, w*np.sqrt(1-lmbda), h*np.sqrt(1-lmbda))
        bbox_l, bbox_t = int(bbox[0]), int(bbox[2])
        bbox_r, bbox_b = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

        x[:, bbox_l:bbox_r, bbox_t:bbox_b] = lmbda * x[:, bbox_l:bbox_r, bbox_t:bbox_b] + (1-lmbda) * x2[:, bbox_l:bbox_r, bbox_t:bbox_b] 
        return to_pil_image(x)