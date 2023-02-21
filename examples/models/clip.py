from transformers import CLIPVisionModel
import torch.nn as nn

"""
Image classification model using CLIP vision_model weights
"""

class CLIPClassifier(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, x):
        hidden_state = super().__call__(
            pixel_values=x,
        )[0] # shape is (B, 257, 1024) for clip-vit-large-patch14
        logits = self.classifier(hidden_state[:, 0, :])
        return logits

class CLIPFeaturizer(CLIPVisionModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        hidden_state = super().__call__(
            pixel_values=x,
        )[0] # shape is (B, 257, 1024) for clip-vit-large-patch14
        return hidden_state[:, 0, :]