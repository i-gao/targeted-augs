from typing import Dict, List

import torch

from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.domain_adversarial_network import ConditionalDomainAdversarialNetwork
from models.initializer import initialize_model
from optimizer import initialize_optimizer_with_model_params
from losses import initialize_loss

class CDAN(SingleModelAlgorithm):
    """
    Conditional domain-adversarial training of neural networks.

    Adapted from: https://github.com/facebookresearch/DomainBed/blob/51810e60c01fbfcf8f2db918b882e4445b8b6527/domainbed/algorithms.py

    Original paper:
        @article{long2018conditional,
          title={Conditional adversarial domain adaptation},
          author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
          journal={Advances in neural information processing systems},
          volume={31},
          year={2018}
        }
    """

    def __init__(
        self,
        config,
        d_out,
        grouper,
        loss,
        metric,
        n_train_steps,
        n_domains,
        n_classes,
        group_ids_to_domains,
    ):
        # Initialize model
        featurizer, classifier = initialize_model(
            config, d_out=d_out, is_featurizer=True
        )
        model = ConditionalDomainAdversarialNetwork(featurizer, classifier, n_domains, n_classes, use_multilinear_map=config.dann_multilinear_map)
        
        parameters_to_optimize: List[Dict] = model.get_parameters_with_lr(
            featurizer_lr=config.dann_featurizer_lr,
            classifier_lr=config.dann_classifier_lr,
            discriminator_lr=config.dann_discriminator_lr,
        )
        self.optimizer = initialize_optimizer_with_model_params(config, parameters_to_optimize)
        self.domain_loss = initialize_loss('cross_entropy', config)

        # Initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.group_ids_to_domains = group_ids_to_domains

        # Algorithm hyperparameters
        self.penalty_weight = config.dann_penalty_weight
        self.balance_discriminator_loss = config.dann_class_balance_reweighting

        # Additional logging
        self.return_features = config.save_features
        self.logged_fields.append("classification_loss")
        self.logged_fields.append("domain_classification_loss")

    def process_batch(self, batch):
        """
        Overrides single_model_algorithm.process_batch().
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
                - domains_true (Tensor): true domains for batch
                - domains_pred (Tensor): predicted domains for batch
        """
        # Forward pass
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.grouper.metadata_to_group(metadata).to(self.device)
        domains_true = self.group_ids_to_domains[g].to(self.device)

        y_pred, features, domains_pred = self.model(x, y_true)

        results = {
            "g": g,
            "metadata": metadata,
            "y_true": y_true,
            "y_pred": y_pred,
            "domains_true": domains_true,
            "domains_pred": domains_pred,
        }
        if self.return_features:
            results['features'] = features
        return results

    def objective(self, results):
        classification_loss = self.loss.compute(
            results["y_pred"], results["y_true"], return_dict=False
        )

        if self.is_training:
            domain_classification_loss = self.domain_loss.compute_element_wise(
                results.pop("domains_pred"),
                results.pop("domains_true"),
                return_dict=False,
            )
            if self.balance_discriminator_loss: 
                # if true, the domain_classification_loss may be up to 1/batch_size of the non-rebalanced loss
                # this occurs if the entire batch is one y or one (y,h)
                n = len(domain_classification_loss)
                _, ixs, y_counts = torch.unique(
                    results["y_true"],
                    dim=0,
                    return_inverse=True,
                    return_counts=True,
                )
                weights = 1. / (y_counts[ixs] * n).float()
                domain_classification_loss = (weights * domain_classification_loss).sum()
            else:
                domain_classification_loss = domain_classification_loss.mean()

        else:
            domain_classification_loss = 0.0

        # Add to results for additional logging
        self.save_metric_for_logging(
            results, "classification_loss", classification_loss
        )
        self.save_metric_for_logging(
            results, "domain_classification_loss", domain_classification_loss
        )
        return classification_loss + domain_classification_loss * self.penalty_weight
