import copy
import torch
from tqdm import tqdm
import math

from configs.supported import process_outputs_functions
from wilds.common.utils import MixedY

from utils import save_model, save_pred, get_pred_prefix, get_model_prefix, collate_list, detach_and_clone

def run_epoch(algorithm, dataset, general_logger, epoch, config, train, batch_transform=None):
    if len(dataset['dataset']) == 0:
        print(f"Warning: split {dataset['name']} is empty.")
        return None, None

    if dataset['verbose']:
        general_logger.write(f"\n{dataset['name']}:\n")

    if train:
        algorithm.train()
        torch.set_grad_enabled(True)
    else:
        algorithm.eval()
        torch.set_grad_enabled(False)

    # Not preallocating memory is slower
    # but makes it easier to handle different types of data loaders
    # (which might not return exactly the same number of examples per epoch)
    epoch_y_true = []
    epoch_y_pred = []
    epoch_metadata = []

    # Assert that data loaders are defined for the datasets
    assert 'loader' in dataset, "A data loader must be defined for the dataset."

    batches = dataset['loader']
    if config.progress_bar:
        batches = tqdm(batches)
    last_batch_idx = len(batches)-1
    
    # Using enumerate(iterator) can sometimes leak memory in some environments (!)
    # so we manually increment batch_idx
    batch_idx = 0
    for labeled_batch in batches:
        if train:
            labeled_batch = batch_transform(*labeled_batch) if batch_transform is not None else labeled_batch
            batch_results = algorithm.update(labeled_batch, is_epoch_end=(batch_idx==last_batch_idx))
        else:
            batch_results = algorithm.evaluate(labeled_batch)

        # These tensors are already detached, but we need to clone them again
        # Otherwise they don't get garbage collected properly in some versions
        # The extra detach is just for safety
        # (they should already be detached in batch_results)
        if isinstance(batch_results['y_true'], MixedY): 
            epoch_y_true.append(detach_and_clone(batch_results['y_true'].y1))
        else:
            epoch_y_true.append(detach_and_clone(batch_results['y_true']))
        y_pred = detach_and_clone(batch_results['y_pred'])

        if config.process_outputs_function is not None:
            y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
        epoch_y_pred.append(y_pred)
        epoch_metadata.append(detach_and_clone(batch_results['metadata']))

        if train: 
            effective_batch_idx = (batch_idx + 1) / config.gradient_accumulation_steps
        else: 
            effective_batch_idx = batch_idx + 1

        if train and effective_batch_idx % config.log_every==0:
            log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

        batch_idx += 1

    epoch_y_pred = collate_list(epoch_y_pred)
    epoch_y_true = collate_list(epoch_y_true)
    epoch_metadata = collate_list(epoch_metadata)

    results, results_str = dataset['dataset'].eval(
        epoch_y_pred,
        epoch_y_true,
        epoch_metadata)

    if config.scheduler_metric_split==dataset['split']:
        algorithm.step_schedulers(
            is_epoch=True,
            metrics=results,
            log_access=(not train))

    # log after updating the scheduler in case it needs to access the internal logs
    log_results(algorithm, dataset, general_logger, epoch, math.ceil(effective_batch_idx))

    results['epoch'] = epoch
    dataset['eval_logger'].log(results)
    if dataset['verbose']:
        general_logger.write('Epoch eval:\n')
        general_logger.write(results_str)

    return results, epoch_y_pred


def train(algorithm, datasets, general_logger, config, epoch_offset, best_val_metric, batch_transform=None, val_split='val', train_split='train'):
    """
    Train loop that, each epoch:
        - Steps an algorithm on the datasets[train_split] split
        - Evaluates the algorithm on the datasets[val_split] split
        - Saves models / preds with frequency according to the configs
        - Evaluates on any other specified splits in the configs
    Assumes that the datasets dict contains labeled data.
    """
    for epoch in range(epoch_offset, config.n_epochs):
        general_logger.write('\nEpoch [%d]:\n' % epoch)

        # First run training
        run_epoch(algorithm, datasets[train_split], general_logger, epoch, config, train=True, batch_transform=batch_transform)

        # Then run val split
        val_results, y_pred = run_epoch(algorithm, datasets[val_split], general_logger, epoch, config, train=False)
        if val_results is not None: 
            curr_val_metric = val_results[config.val_metric]
            general_logger.write(f'Validation {config.val_metric}: {curr_val_metric:.3f}\n')
        else:
            curr_val_metric = None

        if curr_val_metric is None:
            is_best=False
        elif best_val_metric is None:
            is_best = True
        else:
            if config.val_metric_decreasing:
                is_best = curr_val_metric < best_val_metric
            else:
                is_best = curr_val_metric > best_val_metric
        if is_best:
            best_val_metric = curr_val_metric
            general_logger.write(f'Epoch {epoch} has the best validation performance so far.\n')

        save_model_if_needed(algorithm, datasets[val_split], epoch, config, is_best, best_val_metric)
        save_pred_if_needed(y_pred, datasets[val_split], epoch, config, is_best)

        # Then run everything else
        if config.evaluate_all_splits:
            additional_splits = [split for split in datasets.keys() if split not in ['train', train_split, val_split]]
        else:
            additional_splits = config.eval_splits
        for split in additional_splits:
            _, y_pred = run_epoch(algorithm, datasets[split], general_logger, epoch, config, train=False)
            save_pred_if_needed(y_pred, datasets[split], epoch, config, is_best)

        general_logger.write('\n')


def evaluate(algorithm, datasets, epoch, general_logger, config, is_best, train_split='train'):
    algorithm.eval()
    torch.set_grad_enabled(False)
    for split, dataset in datasets.items():
        print(f"Evaluating {dataset['name']}")
        if (not config.evaluate_all_splits) and (split not in config.eval_splits):
            continue
        epoch_y_true = []
        epoch_y_pred = []
        epoch_metadata = []
        iterator = tqdm(dataset['loader']) if config.progress_bar else dataset['loader']

        circuit_features = []
        circuit_metadata = []
        i = 1 # save features every 500 batches
        for batch in iterator:
            batch_results = algorithm.evaluate(batch)
            epoch_y_true.append(detach_and_clone(batch_results['y_true']))
            y_pred = detach_and_clone(batch_results['y_pred'])
            if config.process_outputs_function is not None:
                y_pred = process_outputs_functions[config.process_outputs_function](y_pred)
            epoch_y_pred.append(y_pred)
            metadata = detach_and_clone(batch_results['metadata'])
            epoch_metadata.append(metadata)
            features = detach_and_clone(batch_results['features'])
            
            circuit_features.append(features)
            circuit_metadata.append(metadata)

            if i % 500 == 0:
                print(f"Saving features {i}...")
                save_features(collate_list(circuit_features), collate_list(circuit_metadata), i, dataset, config)
                del circuit_features
                circuit_features = []
                circuit_metadata = []
            i+=1

        # also save at end
        if len(circuit_features):
            print(f"Saving final {len(circuit_features)} features...")
            save_features(collate_list(circuit_features), collate_list(circuit_metadata), i, dataset, config)
        del circuit_features

        epoch_y_pred = collate_list(epoch_y_pred)
        epoch_y_true = collate_list(epoch_y_true)
        epoch_metadata = collate_list(epoch_metadata)
        results, results_str = dataset['dataset'].eval(
            epoch_y_pred,
            epoch_y_true,
            epoch_metadata)

        results['epoch'] = epoch
        dataset['eval_logger'].log(results)
        general_logger.write(f'Eval split {split} at epoch {epoch}:\n')
        general_logger.write(results_str)

        # Skip saving train preds, since the train loader generally shuffles the data
        if split != train_split:
            save_pred_if_needed(epoch_y_pred, dataset, epoch, config, is_best, force_save=True)

def log_results(algorithm, dataset, general_logger, epoch, effective_batch_idx):
    if algorithm.has_log:
        log = algorithm.get_log()
        log['epoch'] = epoch
        log['batch'] = effective_batch_idx
        dataset['algo_logger'].log(log)
        if dataset['verbose']:
            general_logger.write(algorithm.get_pretty_log_str())
        algorithm.reset_log()

def save_features(features, metadata, i, dataset, config):
    """Dump features & metadata to disk"""
    if features is None: return
    if not config.save_features: return
    prefix = get_pred_prefix(dataset, config)
    torch.save(features, f'{prefix}features_{i}.pt')
    torch.save(metadata[:, :2], f'{prefix}metadata_{i}.pt')

def save_pred_if_needed(y_pred, dataset, epoch, config, is_best, force_save=False):
    if y_pred is None: return
    if config.save_pred:
        prefix = get_pred_prefix(dataset, config)
        if force_save or (config.save_step is not None and (epoch + 1) % config.save_step == 0):
            save_pred(y_pred, prefix + f'epoch:{epoch}_pred')
        if (not force_save) and config.save_last:
            save_pred(y_pred, prefix + f'epoch:last_pred')
        if config.save_best and is_best:
            save_pred(y_pred, prefix + f'epoch:best_pred')


def save_model_if_needed(algorithm, dataset, epoch, config, is_best, best_val_metric):
    prefix = get_model_prefix(dataset, config)
    if config.save_step is not None and (epoch + 1) % config.save_step == 0:
        save_model(algorithm, epoch, best_val_metric, prefix + f'epoch:{epoch}_model.pth')
    if config.save_last:
        save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:last_model.pth')
    if config.save_best and is_best:
        save_model(algorithm, epoch, best_val_metric, prefix + 'epoch:best_model.pth')
