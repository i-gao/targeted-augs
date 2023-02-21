from wilds.common.metrics.all_metrics import (
    Accuracy,
    MultiTaskAccuracy,
    MSE,
    multiclass_logits_to_pred,
    binary_logits_to_pred,
    MultiTaskAveragePrecision
)

algo_log_metrics = {
    'accuracy': Accuracy(prediction_fn=multiclass_logits_to_pred),
    'mse': MSE(),
    'multitask_accuracy': MultiTaskAccuracy(prediction_fn=multiclass_logits_to_pred),
    'multitask_binary_accuracy': MultiTaskAccuracy(prediction_fn=binary_logits_to_pred),
    'multitask_avgprec': MultiTaskAveragePrecision(prediction_fn=None),
    None: None,
}

process_outputs_functions = {
    'binary_logits_to_pred': binary_logits_to_pred,
    'multiclass_logits_to_pred': multiclass_logits_to_pred,
    None: None,
}

# See algorithms/initializer.py
algorithms = ['ERM', 'deepCORAL', 'IRM', 'DANN', 'CDAN']

# See models/initializer.py
models = ['resnet50', 'efficientnet-b0', 'densenet121', 'clip-vit-large-patch14']

# See optimizer.py
optimizers = ['SGD', 'Adam', 'AdamW', 'Adafactor']

# See scheduler.py
schedulers = ['linear_schedule_with_warmup', 'cosine_schedule_with_warmup', 'ReduceLROnPlateau', 'StepLR', 'FixMatchLR', 'MultiStepLR', 'CosineAnnealingLR']

# See losses.py
losses = ['cross_entropy', 'lm_cross_entropy', 'MSE', 'multitask_bce', 'cross_entropy_logits']
