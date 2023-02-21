import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to

class ERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        featurizer, classifier = initialize_model(config, d_out, is_featurizer=True)
        featurizer = featurizer.to(config.device)
        classifier = classifier.to(config.device)
        model = torch.nn.Sequential(featurizer, classifier)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        # set model components
        self.featurizer = featurizer
        self.classifier = classifier

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        features = self.featurizer(x)
        outputs = self.classifier(features)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'features': features,
            'metadata': metadata,
        }
        return results


    def objective(self, results):
        labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        return labeled_loss