from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class DomainDiscriminator(nn.Sequential):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    Domain discriminator model from
    `"Domain-Adversarial Training of Neural Networks" <https://arxiv.org/abs/1505.07818>`_
    In the original paper and implementation, we distinguish whether the input features come
    from the source domain or the target domain.

    We extended this to work with multiple domains, which is controlled by the n_domains
    argument.

    Args:
        in_feature (int): dimension of the input feature
        n_domains (int): number of domains to discriminate
        hidden_size (int): dimension of the hidden features
        batch_norm (bool): whether use :class:`~torch.nn.BatchNorm1d`.
            Use :class:`~torch.nn.Dropout` if ``batch_norm`` is False. Default: True.
    Shape:
        - Inputs: (minibatch, `in_feature`)
        - Outputs: :math:`(minibatch, n_domains)`
    """

    def __init__(
        self, in_feature: int, n_domains, hidden_size: int = 1024, batch_norm=True
    ):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_domains),
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, n_domains),
            )

    def get_parameters_with_lr(self, lr) -> List[Dict]:
        return [{"params": self.parameters(), "lr": lr}]

class GradientReverseFunction(Function):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    @staticmethod
    def forward(
        ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.0
    ) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)

class RandomizedMultiLinearMap(nn.Module):
    """
    Randomized multilinear map between tensors of the same shape.
    Adapted from https://github.com/thuml/Transfer-Learning-Library
    """

    def __init__(self, f_dim: int, g_dim: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(f_dim, output_dim)
        self.Rg = torch.randn(g_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output

class MultiLinearMap(nn.Module):
    """
    Credit: https://github.com/thuml/Transfer-Learning-Library
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1)

class DomainAdversarialNetwork(nn.Module):
    def __init__(self, featurizer, classifier, n_domains):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains)
        self.gradient_reverse_layer = GradientReverseLayer()

    def forward(self, input):
        features = self.featurizer(input)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        domains_pred = self.domain_classifier(features)
        return y_pred, features, domains_pred

    def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
        ]
        return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)

class ConditionalDomainAdversarialNetwork(nn.Module):
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library/blob/6e5335ba9c5a101a09269b28d2b8c8f8500223eb/tllib/alignment/cdan.py
    and https://github.com/facebookresearch/DomainBed/blob/51810e60c01fbfcf8f2db918b882e4445b8b6527/domainbed/algorithms.py 
    """
    def __init__(self, featurizer, classifier, n_domains, n_classes, use_multilinear_map=False):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.domain_classifier = DomainDiscriminator(featurizer.d_out, n_domains)
        self.gradient_reverse_layer = GradientReverseLayer()
        self.embeddings = nn.Embedding(n_classes, featurizer.d_out)
        if use_multilinear_map:
            self.map = RandomizedMultiLinearMap(featurizer.d_out)
        else:
            self.map = lambda x,y: x + y

    def forward(self, input, y):
        features = self.featurizer(input)
        y_pred = self.classifier(features)
        features = self.gradient_reverse_layer(features)
        domains_pred = self.domain_classifier(self.map(features, self.embeddings(y)))
        return y_pred, features, domains_pred

    def get_parameters_with_lr(self, featurizer_lr, classifier_lr, discriminator_lr) -> List[Dict]:
        """
        Adapted from https://github.com/thuml/Transfer-Learning-Library

        A parameter list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer
        """
        # In TLL's implementation, the learning rate of this classifier is set 10 times to that of the
        # feature extractor for better accuracy by default. For our implementation, we allow the learning
        # rates to be passed in separately for featurizer and classifier.
        params = [
            {"params": self.featurizer.parameters(), "lr": featurizer_lr},
            {"params": self.classifier.parameters(), "lr": classifier_lr},
            {"params": self.embeddings.parameters(), "lr": discriminator_lr},
        ]
        return params + self.domain_classifier.get_parameters_with_lr(discriminator_lr)
