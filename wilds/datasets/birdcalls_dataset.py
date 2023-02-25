from pathlib import Path
import librosa

from PIL import Image
import pandas as pd
import numpy as np
import torch
import ast
from collections import defaultdict

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1


class BirdCallsDataset(WILDSDataset):
    """
    An ornithology dataset of bird calls from Hawaii, the Southwest Amazon Basin, and Ithaca, NY.
    This is a modified version of three ornithology datasets:
        https://zenodo.org/record/7078499#.Y1fRcHbMJaQ (Hawaii)
        https://zenodo.org/record/7079124#.Y1fRlXbMJaQ (Amazon)
        https://zenodo.org/record/7079380 (Ithaca)
    
    Supported `split_scheme`:
        - 'combined'
    Input (x):
        5-second audio clip, converted to an RGB image of a mel-spectrogram over frequencies 0-16kHz
    Label (y):
        y is one of 32 classes, with 0 referring to an empty clip (no bird observed)
    Metadata:
        Each example is annotated with the ID of the microphone it came from, which we refer to as the 'location.'
        Each example also has an expert-annotated time-frequency bounding box for the bird call.

    License:
        This dataset is distributed under Creative Commons 4.0 International

    When keep_ood_train is False, examples from the splits 'ood_train', 'ood_val', and 'test' are concatenated into the 'test' split.
    We set keep_ood_train to True for test-to-test ablations in Appendix A.3.
    """
    _dataset_name = 'birdcalls'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x83846a709fbc4cf08ad14bee87ec8193/contents/blob/',
            'compressed_size': 11_225_304_127}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='hawaii-234', keep_ood_train=False):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme not in  ('combined'):
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path; download main dataset from codalab bundle
        # bboxes and segmentations masks are expected to be found in root_dir/, not root_dir/iwildcam_v_x
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        
        # Load splits
        df = pd.read_csv(self._data_dir / f'{self._split_scheme}.csv')

        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4, 'ood_train': 5, 'ood_val': 6}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)', 'ood_train': 'Train (OOD/Trans)', 'ood_val': 'Validation (OOD/Trans)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        if not keep_ood_train: df.loc[df['split_id'].isin([5, 6]), 'split_id'] = 2
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['mel_img_path'].values
        self._audio_array = df['audio_path'].values

        # Labels
        self.y_names = {**{y: i+1 for i, y in enumerate(df['Species eBird Code'].unique()) if y != 'empty'}, **{'empty': 0}}
        self._y_array = torch.tensor(df['Species eBird Code'].apply(lambda x: self.y_names[x]).values)
        self._n_classes = max(self._y_array) + 1
        self._y_size = 1
        
        # Location/group info
        n_groups = len(df['Location'].unique())
        self._n_groups = n_groups
        self.location_array = torch.tensor(df['Location'].values)

        if 'dataset' in df.columns:
            dataset_map = {v: i for i,v in enumerate(df['dataset'].unique())}
            self.cluster_array = torch.tensor(df['dataset'].apply(lambda v: dataset_map[v]).values)
        else:
            self.cluster_array = torch.ones(len(self.y_array))
        
        self._metadata_array = torch.tensor(np.stack([df['Location'].values,
                            df['Call Low Freq (Hz)'].values, df['Call High Freq (Hz)'].values,
                            self.cluster_array, self.y_array], axis=1))
        self._metadata_fields = ['location', 'low_freq', 'high_freq', 'cluster', 'y']

        # additional info for the copy-paste augmentation
        self.img_ids = df.id.values   
        self.bboxes = df['mel_bbox'].apply(lambda s: ast.literal_eval(s) if type(s) == str else s)  
        self.empty_indices = {
            'all': np.where(self._y_array == 0)[0],
            **{k: np.where(
                (self._y_array == 0) & (self._split_array == v)
            )[0] for k,v in self._split_dict.items()}
        }
        self.y_to_observed_locs = defaultdict(torch.Tensor, {y.item(): torch.unique(self.location_array[self._y_array == y]) for y in torch.unique(self._y_array)})

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['y']),
        )

        super().__init__(root_dir, download, split_scheme)

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
                        })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n"
        )

        return results, results_str

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (PIL image): image of the idx-th data point
        """
        img_path = self.data_dir / self._input_array[idx]
        img = Image.open(img_path)
        return img

    def get_audio(self, idx, normalize=False):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (np array): waveform for the idx-th data point
            - sr (int): sample rate
        """
        f = self.data_dir / self._audio_array[idx]
        sample_rate = librosa.get_samplerate(f)
        x, sr = librosa.load(f, sr=sample_rate)
        if normalize: x = librosa.util.normalize(x)
        return x, sr

    def get_bbox(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - bboxes (list of lists): list of bboxes
                bboxes are [x, y, w, h] with (x,y) being the top left of the box
            - conf (float): max confidence over the bboxes (because this is human-annotated, conf=1. we need to return
                two values for the cutpaste augmentation)
        """
        bbox = self.bboxes[idx]
        if type(bbox) != list: return [[]], 0
        else: return [bbox], 1
