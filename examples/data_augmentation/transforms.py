import copy
from typing import List

import numpy as np
import torch
import torchvision.transforms as transforms

from data_augmentation.randaugment import FIX_MATCH_AUGMENTATION_POOL, RandAugment
from data_augmentation.cutout import Cutout
from data_augmentation.copy_paste import CopyPasteAugment
from data_augmentation.stain_color_jitter import StainColorJitter
from data_augmentation.random_crop import RandomCrop
from data_augmentation.bg_challenge import BGChallenge
from data_augmentation.lisa import LISAMixUp, LISACutMix
from data_augmentation.audio import NoiseReduceAugment, SpecAugment, RandomPass

from utils import is_nested_list, _move_to_front


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]

def initialize_transform(
    config, dataset, is_training, grouper=None, base_transforms=None, additional_transforms=None, transform_p=1.0,
):
    """
    Args:
        - base_transforms, additional_transforms: str
    """
    has_bboxes = (dataset.dataset_name in ["iwildcam", "birdcalls"])
    is_birds = (dataset.dataset_name == "birdcalls")
    is_image = (dataset.dataset_name in ['camelyon17', 'iwildcam', 'birdcalls'])

    # input type checking
    if not is_nested_list(base_transforms): base_transforms = _parse_transform_str(base_transforms)
    if not is_nested_list(additional_transforms): additional_transforms = _parse_transform_str(additional_transforms)
    assert type(base_transforms) == list and type(additional_transforms) == list

    # ensure that len(base_transforms) == len(additional_transforms)
    # i.e. we have the same number of branches
    base_transforms, additional_transforms, n_branches = _check_transforms(base_transforms, additional_transforms)

    # layer on transforms in given order
    branches = []
    for i in range(n_branches):
        # decide the order in which to apply the augmentations
        branch_transforms = base_transforms[i] + additional_transforms[i]
        use_input_dropout = ('input_dropout' in branch_transforms)
        if use_input_dropout:
            assert config.input_dropout_p is not None and config.input_dropout_p <= 1
            branch_transforms.remove('input_dropout')
        if has_bboxes:
            print("By default, base_transforms are applied before additional_transforms, but resizing is not applied to the bboxes. So move transforms that use bboxes to be the FIRST transforms applied, followed by resizing.")
            branch_transforms = _move_to_front(branch_transforms, special_items=('copypaste', 'copypaste_same_region', 'randaugment', 'random_crop', 'only_bg_b', 'no_fg', 'only_fg'))

        # get the transforms
        branch_steps = []
        for transform_name in branch_transforms:
            try:
                init_fn = globals()[f"initialize_{transform_name}_transform"]
            except:
                raise ValueError(f"{transform_name} not recognized")
            steps = init_fn(config, dataset, is_training, grouper)
            if type(steps) == list:
                branch_steps.extend([Transform(t, apply_stochastically=(transform_name in additional_transforms[i])) for t in steps])
            else:
                branch_steps.append(Transform(steps, apply_stochastically=(transform_name in additional_transforms[i])))

        # postprocessing: convert images to tensors if appropriate
        if config.to_tensor and is_image:
            branch_steps.append(Transform(transforms.ToTensor()))

        # add image normalization if appropriate
        if config.to_tensor and is_image and not is_birds:
            branch_steps.append(Transform(
                transforms.Normalize(
                    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
                    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
                )
            ))

        if use_input_dropout: branch_steps.append(Transform(
            torch.nn.functional.dropout, p=config.input_dropout_p, apply_stochastically=False # already stochastic
        ))

        branches.append(SequentialTransforms(steps=branch_steps, p=transform_p))

    augmentation = ParallelTransforms(branches)
    print(augmentation)

    return augmentation

#################

class SequentialTransforms:
    """Custom version of transforms.Compose s.t. we can apply non-consecutive transformations stochastically."""
    def __init__(self, steps, p):
        self.transforms = steps
        self.transforms[-1].last_in_chain = True
        self.p = p # probability of applying the chain

    def __call__(self, input):
        random_apply =  (self.p >= torch.rand(1)).item()
        for transform in self.transforms:
            if (random_apply == True) or (transform.apply_stochastically == False):
                input = transform(input)
        return input

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    p={self.p}"
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class ParallelTransforms:
    """
    Instead of returning a single transform(x), return transform1(x) or transform2(x) with uniform probability
    """
    def __init__(self, branches):
        self.branches = branches

    def __call__(self, input):
        return np.random.choice(self.branches)(input)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.branches:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class Transform:
    """"
    Wrapper for a callable transform t.
    Goal: in the rest of the code, call transforms on tuple, i.e. Transform(t).__call__((x, b))
        This allows us to pass in b=y or b=ix, and to pass in other arguments to the __call__ fn.
    However, since not all t are supposed to take in 2 args (i.e., the torchvision transforms only take x),
    this class handles the underlying logic of whether to call t(x) or t(x,b).
    """
    def __init__(self, transform_obj, last_in_chain=False, apply_stochastically=False, **kwargs):
        self.transform = transform_obj
        if hasattr(transform_obj, 'nargs'):
            self.nargs = transform_obj.nargs # supply a tuple to __call__ fn, e.g. to pass in (x, ix) or (x,y)
        else:
            self.nargs = 1 # supply just x to __call__ fn
        self.last_in_chain = last_in_chain
        self.apply_stochastically = apply_stochastically
        self.kwargs = kwargs

    def __call__(self, input):
        img = self.transform(*input[:self.nargs], **self.kwargs)
        if self.last_in_chain:
            return img
        else:
            return (
                img, # img
                *input[1:], # other args
            )

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"({str(self.transform)}, stochastic={self.apply_stochastically})"
        return format_string

def _parse_transform_str(v):
    """
    Input: string specifying the transforms to apply
        sequential transforms are separated by spaces
        parallel transforms are separated by colons
    Output: [[str]], where the outerlist specifies the branches in ParallelTransforms, 
    while the inner list is a list of strings that is passed into SequentialTransforms.
    Examples:
        run two transforms in sequence:
            "image_base randaugment" => [['image_base', 'randaugment']]
        run two transforms in parallel, picking the branch with random probability:
            "image_base:randaugment" => ['image_base', ['randaugment']]
    """
    if v is None: return [[]]
    if type(v) == str: v = [v]
    transforms = ' '.join(v).split(':')
    transforms = [t.strip().split(' ') for t in transforms]
    if not is_nested_list(transforms): transforms = [transforms]
    return transforms

def _check_transforms(base_transforms, additional_transforms):
    """
    Enforce that the number of branches in base_transforms and additional_transforms are the same.
    """
    # error checking
    if len(additional_transforms) == len(base_transforms):
        n_branches = len(additional_transforms)
    elif len(additional_transforms) > 1: # more than one branch specified by additional_transforms
        assert len(base_transforms) == 1, "Cannot broadcast number of base_transform branches to the number of additional_transform branches."
        n_branches = len(additional_transforms)
        base_transforms = [base_transforms[0]] * n_branches
    elif len(base_transforms) > 1: # more than one branch specified by base_transforms, but only one branch by additional_transforms
        assert len(additional_transforms) == 1, "Cannot broadcast number of additional_transform branches to the number of base_transform branches." # no-op
        n_branches = len(base_transforms)
        additional_transforms = [additional_transforms[0]] * n_branches
    else: 
        raise ValueError("Both base_transform and additional_transform should have at least one branch, i.e. be nested lists containing at least 1 inner list, which can be empty.")
    return base_transforms, additional_transforms, n_branches

def _get_target_resolution(config, dataset):
    if config.target_resolution is not None:
        return config.target_resolution
    else:
        return dataset.original_resolution


#################################################################################
#### TRANSFORMS: function names should be intialize_*_transform #################
#################################################################################

def initialize_image_base_transform(config, dataset, is_training, grouper) -> List:
    transform_steps = []

    if dataset.original_resolution is not None and min(
        dataset.original_resolution
    ) != max(dataset.original_resolution):
        crop_size = min(dataset.original_resolution)
        transform_steps.append(transforms.CenterCrop(crop_size)) # this is a no-op on iwildcam!

    if config.target_resolution is not None:
        transform_steps.append(transforms.Resize(config.target_resolution))

    return transform_steps

def initialize_randaugment_transform(config, dataset, is_training, grouper) -> List:
    """
    Pure RandAugment transform
    Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    """
    return [
        RandAugment(
            n=config.transform_kwargs['randaugment_n'],
            augmentation_pool=FIX_MATCH_AUGMENTATION_POOL,
        ),
        transforms.RandomHorizontalFlip(), # need to do random flipping last, s.t. bounding boxes aren't messed up
    ]

def initialize_colorjitter_transform(config, dataset, is_training, grouper) -> List:
    """Jitter saturation and hue only."""
    return [
        transforms.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=2,
            hue=0.5,
        )
    ]

def initialize_cutout_transform(config, dataset, is_training, grouper) -> List:
    """
    Pure Cutout transform
    Adapted from https://github.com/YBZh/Bridging_UDA_SSL
    """
    return [
        Cutout(
            labeled_dataset=dataset,
            bbox_aware=False
        ),
        transforms.RandomHorizontalFlip(), # need to do random flipping last, s.t. bounding boxes aren't messed up
    ]

def initialize_aware_cutout_transform(config, dataset, is_training, grouper) -> List:
    """
    Cutout w/ bbox checking
    """
    return [
        Cutout(
            labeled_dataset=dataset,
            bbox_aware=True
        ),
        transforms.RandomHorizontalFlip(), # need to do random flipping last, s.t. bounding boxes aren't messed up
    ]

def initialize_random_crop_transform(config, dataset, is_training, grouper) -> List:
    target_resolution = _get_target_resolution(config, dataset)
    return [
        RandomCrop(
            size=target_resolution,
            labeled_dataset=dataset,
            bbox_aware=True
        ),
    ]

def initialize_lisa_transform(config, dataset, is_training, grouper) -> List:
    return [
        LISAMixUp(dataset=dataset, grouper=grouper, **config.transform_kwargs),
    ]

def initialize_lisa_cutmix_transform(config, dataset, is_training, grouper) -> List:
    return [
        LISACutMix(dataset=dataset, grouper=grouper, **config.transform_kwargs),
    ]

### Camelyon17 Transforms ###
def initialize_camelyon_color_transform(config, dataset, is_training, grouper) -> List:
    # The color transform needs to come after ToTensor, which is unfortunately added at the end by default
    # so need to convert back to PIL
    return [
        transforms.ToTensor(),
        StainColorJitter(
            **config.transform_kwargs
        ),
        transforms.ToPILImage(),
    ]

## BirdCalls Transforms ###

def initialize_noisereduce_transform(config, dataset, is_training, grouper) -> List:
    return [
        NoiseReduceAugment(dataset, **config.transform_kwargs),
    ]

def initialize_specaugment_transform(config, dataset, is_training, grouper) -> List:
    return [
        SpecAugment(dataset, **config.transform_kwargs),
    ]

def initialize_randompass_transform(config, dataset, is_training, grouper) -> List:
    return [
        RandomPass(dataset, **config.transform_kwargs),
    ]

## Copy-Paste ##

def initialize_copypaste_transform(config, dataset, is_training, grouper) -> List:
    return [
        CopyPasteAugment( # DO NOT FLIP OR CROP BEFORE THIS BECAUSE BBOXES & MASKS WILL GET MESSED UP
            dataset, **config.transform_kwargs
        ),
        transforms.RandomHorizontalFlip(),
    ]


def initialize_copypaste_same_region_transform(config, dataset, is_training, grouper) -> List:
    print("Warning: using a named CopyPaste transform, so ignoring config.transform_kwargs")
    return [
        CopyPasteAugment( # DO NOT FLIP OR CROP BEFORE THIS BECAUSE BBOXES & MASKS WILL GET MESSED UP
            dataset,
            same_cluster=True,
        ),
        transforms.ColorJitter(
            brightness=0,
            contrast=0,
            saturation=2,
            hue=0.5,
        ),
        transforms.RandomHorizontalFlip(),
    ]


def initialize_copypaste_same_y_transform(config, dataset, is_training, grouper) -> List:
    print("Warning: using a named CopyPaste transform, so ignoring config.transform_kwargs")
    return [
        CopyPasteAugment( # DO NOT FLIP OR CROP BEFORE THIS BECAUSE BBOXES & MASKS WILL GET MESSED UP
            dataset,
            same_y_observed=True,
        ),
        transforms.RandomHorizontalFlip(),
    ]


## Background Challenge (Xiao et al. 2020) transforms ##

def initialize_only_bg_b_transform(config, dataset, is_training, grouper) -> List:
    return [
        BGChallenge( # DO NOT FLIP OR CROP BEFORE THIS BECAUSE BBOXES & MASKS WILL GET MESSED UP
            dataset, mode='only-bg-b'
        )
    ]

def initialize_no_fg_transform(config, dataset, is_training, grouper) -> List:
    return [
        BGChallenge( # DO NOT FLIP OR CROP BEFORE THIS BECAUSE BBOXES & MASKS WILL GET MESSED UP
            dataset, mode='no-fg'
        )
    ]

def initialize_only_fg_transform(config, dataset, is_training, grouper) -> List:
    return [
        BGChallenge( # DO NOT FLIP OR CROP BEFORE THIS BECAUSE BBOXES & MASKS WILL GET MESSED UP
            dataset, mode='only-fg'
        )
    ]
