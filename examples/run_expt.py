import os
import argparse
import pandas as pd
import torch
import glob
import sys
from collections import defaultdict

try:
    import wandb
except Exception as e:
    pass

# use local wilds package
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool
from train import train, evaluate
from algorithms.initializer import initialize_algorithm
from data_augmentation.transforms import initialize_transform, _parse_transform_str
from data_augmentation.batch_transform import initialize_batch_transform
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing

# Necessary for large images of GlobalWheat-WILDS
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():

    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')

    # Transforms
    parser.add_argument('--eval_base_transforms', default=None, type=_parse_transform_str, help='Names of transforms applied to eval examples, and never applied stochastically')
    parser.add_argument('--train_base_transforms', default=None, type=_parse_transform_str, help='Names of transforms applied to train examples, and never applied stochastically')
    parser.add_argument('--train_additional_transforms', default=None, type=_parse_transform_str, help='Names of additional transforms applied on top of base transforms to the train examples, and applied stochastically with probability transform_p')
    parser.add_argument('--transform_p', type=float, default=1.0, help='probability w/ which to apply train_additional_transforms')
    parser.add_argument('--transform_kwargs', nargs='*', action=ParseKwargs, default={}, help='transform-specific arguments, e.g. randaugment_n=2')
    parser.add_argument('--batch_transform', default=None, help='name of MixUp / CutMix style augmentation applied over batches as a unit')
    parser.add_argument('--batch_transform_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--input_dropout_p', type=float)
    parser.add_argument('--to_tensor', type=bool, default=True, help="If true, converts images to tensors. If false, leaves images as PIL images.")

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--dann_class_balance_reweighting', default=False, type=parse_bool, const=True, nargs='?', help='If true, reweight the discriminator loss function to account for class imbalance.')
    parser.add_argument('--dann_use_habitats', default=False, type=parse_bool, const=True, nargs='?', help='If true, conditions DANN on Y and cluster.')
    parser.add_argument('--dann_multilinear_map', default=False, type=parse_bool, const=True, nargs='?', help='If true, conditions DANN on Y (and cluster) using a randomized multilinear map with features rather than addition.')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--algo_log_metric')

    # Model selection
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--val_split', default='val')
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_features', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device("cuda")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    # Initialize logs
    if os.path.exists(config.log_dir) and config.resume:
        resume=True
        mode='a'
    elif os.path.exists(config.log_dir) and config.eval_only:
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.seed)

    # Data
    full_dataset = wilds.get_dataset(
        dataset=config.dataset,
        version=config.version,
        root_dir=config.root_dir,
        download=config.download,
        split_scheme=config.split_scheme,
        **config.dataset_kwargs)

    # Initialize grouper
    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.groupby_fields
    )
    
    # Transforms & data augmentations for labeled dataset
    # See transform.py for details
    train_transform = initialize_transform(
        base_transforms=config.train_base_transforms,
        additional_transforms=config.train_additional_transforms,
        transform_p=config.transform_p,
        config=config,
        dataset=full_dataset,
        grouper=train_grouper,
        is_training=True)
    eval_transform = initialize_transform(
        base_transforms=config.eval_base_transforms,
        config=config,
        dataset=full_dataset,
        grouper=train_grouper,
        is_training=False)

    # BatchTransforms like MixUp operate on a batch at a time
    # we'll pass this into train(), which will call this transform
    # in the for batch in the train loop
    batch_transform = initialize_batch_transform(
        batch_transform_name=config.batch_transform,
        config=config, 
        dataset=full_dataset,
        batch_transform_kwargs=config.batch_transform_kwargs,
        transform_p=config.transform_p)

    # Configure labeled torch datasets (WILDS dataset splits)
    datasets = defaultdict(dict)
    for split in full_dataset.split_dict.keys():
        if split == config.train_split:
            transform = train_transform
            verbose = True
        elif split == config.val_split:
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.frac,
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=datasets[split]['dataset'],
                batch_size=config.batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.n_groups_per_batch,
                **config.loader_kwargs)
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.eval_loader,
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.batch_size,
                **config.loader_kwargs)

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
        )
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
        )

    if config.use_wandb:
        initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.no_group_logging:
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    # Initialize algorithm & load pretrained weights if provided
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper,
    )

    if not config.eval_only:
        # Resume from most recent model in log_dir
        resume_success = False
        if resume:
            save_path = glob.glob(config.log_dir + '/*last_model.pth')
            save_path = save_path[0] if len(save_path) else None
            if save_path is None:
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = glob.glob(config.log_dir + f'/*epoch:{latest_epoch}_model.pth')
                    save_path = save_path[0] if len(save_path) else None
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path, device=config.device)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass
        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        # Log effective batch size
        if config.gradient_accumulation_steps > 1:
            logger.write(
                (f'\nUsing gradient_accumulation_steps {config.gradient_accumulation_steps} means that')
                + (f' the effective labeled batch size is {config.batch_size * config.gradient_accumulation_steps}')
                + ('. Updates behave as if torch loaders have drop_last=False\n')
            )

        train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric,
            batch_transform=batch_transform,
            val_split=config.val_split,
            train_split=config.train_split,
        )
    else:
        if config.eval_epoch is None:
            eval_model_path = glob.glob(config.log_dir + '/*best_model.pth')
            eval_model_path = eval_model_path[0] if len(eval_model_path) else None
        else:
            eval_model_path = glob.glob(config.log_dir + f'/*epoch:{config.eval_epoch}_model.pth')
            eval_model_path = eval_model_path[0] if len(eval_model_path) else None
        best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
        if config.eval_epoch is None:
            epoch = best_epoch
        else:
            epoch = config.eval_epoch
        if epoch == best_epoch:
            is_best = True
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config,
            is_best=is_best)

    if config.use_wandb:
        wandb.finish()
    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()

if __name__=='__main__':
    main()
