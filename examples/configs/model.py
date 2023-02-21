model_defaults = {
    'densenet121': {},
    'efficientnet-b0': {},
    'resnet50': {},
    'clip-vit-large-patch14': {
        'optimizer': 'AdamW',
        'model_kwargs': {'pretrained': True},
        'target_resolution': (224, 224),
        'scheduler': 'CosineAnnealingLR',
        'batch_size': 9,
        'gradient_accumulation_steps': 14, # effective batch size of 126
    }
}
