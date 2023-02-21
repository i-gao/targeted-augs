import torch
import torch.nn as nn
import os
import traceback

from models.layers import Identity
from utils import load, getattr_recursive, setattr_recursive

def initialize_model(config, d_out, is_featurizer=False):
    """
    Initializes models according to the config
        Args:
            - config (dictionary): config dictionary
            - d_out (int): the dimensionality of the model output
            - is_featurizer (bool): whether to return a model or a (featurizer, classifier) pair that constitutes a model.
        Output:
            If is_featurizer=True:
            - featurizer: a model that outputs feature Tensors of shape (batch_size, ..., feature dimensionality)
            - classifier: a model that takes in feature Tensors and outputs predictions. In most cases, this is a linear layer.

            If is_featurizer=False:
            - model: a model that is equivalent to nn.Sequential(featurizer, classifier)

        Pretrained weights are loaded according to config.pretrained_model_path using either transformers.from_pretrained (for bert-based models)
        or our own utils.load function (for torchvision models, resnet18-ms, and gin-virtual).
        There is currently no support for loading pretrained weights from disk for other models.
    """
    # If requested, split into (featurizer, classifier) for the purposes of loading only the featurizer,
    # before recombining them at the end
    featurize = is_featurizer

    if config.model in ('resnet50', 'densenet121', 'efficientnet-b0'):
        if featurize:
            featurizer = initialize_torchvision_model(
                name=config.model,
                d_out=None,
                **config.model_kwargs)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_torchvision_model(
                name=config.model,
                d_out=d_out,
                **config.model_kwargs)

    elif config.model == 'logistic_regression':
        assert not featurize, "Featurizer not supported for logistic regression"
        model = nn.Linear(out_features=d_out, **config.model_kwargs)
    
    elif 'clip' in config.model:
        image_size = config.target_resolution[0]
        if featurize:
            featurizer = initialize_clip_based_model(config, d_out, image_size, featurize)
            classifier = nn.Linear(featurizer.d_out, d_out)
            model = (featurizer, classifier)
        else:
            model = initialize_clip_based_model(config, d_out, image_size)

    else:
        raise ValueError(f'Model: {config.model} not recognized.')

    # Load pretrained weights from disk using our utils.load function
    if config.pretrained_model_path is not None:
        if config.model in ('code-gpt-py', 'logistic_regression', 'unet-seq'):
            # This has only been tested on some models (mostly vision), so run this code iff we're sure it works
            raise NotImplementedError(f"Model loading not yet tested for {config.model}.")

        if 'bert' not in config.model:  # We've already loaded pretrained weights for bert-based models using the transformers library
            try:
                if featurize:
                    model_to_load = nn.Sequential(*model)
                else:
                    model_to_load = model

                prev_epoch, best_val_metric = load(
                    model_to_load,
                    config.pretrained_model_path,
                    device=config.device)

                print(
                    (f'Initialized model with pretrained weights from {config.pretrained_model_path} ')
                    + (f'previously trained for {prev_epoch} epochs ' if prev_epoch else '')
                    + (f'with previous val metric {best_val_metric} ' if best_val_metric else '')
                )
            except Exception as e:
                print('Something went wrong loading the pretrained model:')
                traceback.print_exc()
                raise

    # Recombine model if we originally split it up just for loading
    if featurize and not is_featurizer:
        model = nn.Sequential(*model)

    # The `needs_y` attribute specifies whether the model's forward function
    # needs to take in both (x, y).
    # If False, Algorithm.process_batch will call model(x).
    # If True, Algorithm.process_batch() will call model(x, y) during training,
    # and model(x, None) during eval.
    if not hasattr(model, 'needs_y'):
        # Sometimes model is a tuple of (featurizer, classifier)
        if is_featurizer:
            for submodel in model:
                submodel.needs_y = False
        else:
            model.needs_y = False

    return model

def initialize_clip_based_model(config, d_out, image_size, featurize=False):
    from models.clip import CLIPClassifier, CLIPFeaturizer

    if config.pretrained_model_path:
        print(f'Initialized model with pretrained weights from {config.pretrained_model_path}')
        config.model_kwargs['state_dict'] = torch.load(config.pretrained_model_path, map_location=config.device)

    pretrained = config.model_kwargs.pop('pretrained')
    if pretrained:
        image_size_kwargs = {'image_size': image_size, 'ignore_mismatched_sizes': True} if image_size is not None else {}
        if featurize:
            model = CLIPFeaturizer.from_pretrained(
                "openai/" + config.model,
                **image_size_kwargs,
                **config.model_kwargs)
        else:
            model = CLIPClassifier.from_pretrained(
                "openai/" + config.model,
                num_labels=d_out,
                **image_size_kwargs,
                **config.model_kwargs)
    else:
        raise ValueError("In this codebase, CLIP must be instantiated from pretrained.")
    return model

def initialize_torchvision_model(name, d_out, **kwargs):
    import torchvision

    # get constructor and last layer names
    if name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name == 'resnet50':
        constructor_name = name
        last_layer_name = 'fc'
    elif name == 'efficientnet-b0':
        constructor_name = 'efficientnet_b0'
        last_layer_name = 'classifier.1'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d_features = getattr_recursive(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d_features)
        model.d_out = d_features
    else: # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d_features, d_out)
        model.d_out = d_out
    setattr_recursive(model, last_layer_name, last_layer)
    return model
