from datetime import datetime
from pathlib import Path
import os

from PIL import Image
import pandas as pd
import numpy as np
import torch
import json
from torchvision.transforms import functional as F
import time
from collections import defaultdict

from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

# sample of 500 bboxes has mean height >= 20% of the frame or not
BIG_ANIMALS = [1, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 52, 53, 54, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 78, 81, 82, 87, 88, 89, 90, 93, 94, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 109, 111, 112, 113, 114, 115, 116, 118, 119, 125, 126, 127, 133, 134, 135, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 167, 168, 172, 174, 175, 179, 180]
SMALL_ANIMALS = [128, 129, 2, 3, 130, 131, 132, 136, 9, 137, 11, 138, 139, 14, 140, 141, 17, 18, 148, 27, 156, 31, 160, 33, 161, 162, 36, 163, 38, 164, 165, 166, 169, 43, 170, 171, 46, 173, 176, 177, 50, 51, 178, 181, 55, 61, 75, 76, 79, 80, 83, 84, 85, 86, 91, 92, 95, 98, 105, 108, 110, 117, 120, 121, 122, 123, 124]

# mean hour is w/n working hours 9a-5p v. not
DAY_ANIMALS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 46, 48, 49, 51, 52, 53, 54, 56, 58, 59, 60, 61, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 126, 127, 130, 132, 133, 135, 136, 137, 138, 139, 140, 141, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 157, 159, 161, 162, 163, 164, 165, 166, 168, 170, 171, 172, 173, 174, 176, 177, 181]
NIGHT_ANIMALS = [128, 129, 131, 134, 9, 11, 142, 143, 17, 19, 22, 154, 156, 29, 158, 160, 38, 167, 169, 45, 47, 175, 50, 178, 179, 180, 55, 57, 62, 63, 66, 67, 79, 83, 88, 91, 92, 94, 110, 122, 124, 125]

# clusters by geocoordinates
# CLUSTERS[0] are the unknown geocoordinates
# note: completely excludes unlabeled cameras (for which we have zilch geocoordinates)
CLUSTERS = [
    [7, 8, 9, 11, 16, 17, 18, 19, 20, 22, 28, 31, 32, 34, 36, 37, 38, 39, 40, 43, 44, 45, 46, 49, 51, 59, 62, 69, 75, 79, 80, 83, 87, 88, 89, 93, 94, 98, 99, 103, 106, 107, 109, 111, 112, 114, 116, 119, 129, 130, 132, 139, 143, 144, 146, 147, 149, 150, 151, 153, 155, 157, 158, 159, 165, 170, 172, 174, 177, 181, 182, 185, 187, 188, 190, 191, 192, 194, 195, 196, 197, 199, 202, 203, 204, 212, 214, 216, 218, 222, 224, 226, 228, 229, 238, 239, 242, 244, 245, 248, 251, 254, 257, 258, 263, 264, 265, 266, 267, 268, 269, 272, 273, 274, 276, 277, 279, 283, 284, 285, 292, 297, 299, 305, 306, 308, 313, 314, 317, 318, 319, 321],
    [0, 13, 33, 71, 92, 113, 122, 123, 164, 173, 186, 200, 220, 236, 281, 291, 298, 303, 310, 316, 66, 81, 175, 217, 58, 95, 127, 156, 237],
    [1, 5, 10, 12, 14, 15, 23, 30, 35, 41, 47, 50, 52, 53, 60, 61, 65, 67, 68, 70, 72, 74, 77, 85, 90, 91, 96, 97, 105, 110, 121, 124, 128, 131, 133, 135, 137, 138, 140, 141, 142, 152, 161, 166, 167, 178, 180, 198, 201, 205, 206, 208, 209, 213, 215, 219, 225, 231, 232, 233, 234, 235, 247, 252, 256, 271, 290, 293, 300, 304, 312, 322, 108, 136, 154, 171, 183, 211, 275, 24, 56, 73, 76, 78, 86, 104, 115, 125, 163, 169, 184, 241, 280, 282, 287, 288, 302, 315],
    [2, 6, 26, 48, 54, 63, 84, 102, 117, 118, 160, 162, 168, 179, 189, 221, 223, 230, 243, 249, 253, 255, 259, 262, 286, 294, 295, 296, 307, 3, 27, 57, 134, 260, 309, 320, 101, 120, 270, 289, 301, 311],
    [4, 25, 82, 126, 145, 148, 210, 227, 261, 21, 176, 193, 207],
    [42, 64, 100, 246, 250, 278, 29, 240],
    [55],
]


class IWildCamDataset(WILDSDataset):
    """
        The iWildCam2020 dataset.
        This is a modified version of the original iWildCam2020 competition dataset.
        Supported `split_scheme`:
            - 'official'
        Input (x):
            RGB images from camera traps
        Label (y):
            y is one of 186 classes corresponding to animal species
        Metadata:
            Each image is annotated with the ID of the location (camera trap) it came from.
        Website:
            https://www.kaggle.com/c/iwildcam-2020-fgvc7
        Original publication:
            @article{beery2020iwildcam,
            title={The iWildCam 2020 Competition Dataset},
            author={Beery, Sara and Cole, Elijah and Gjoka, Arvi},
            journal={arXiv preprint arXiv:2004.10340},
                    year={2020}
            }
        License:
            This dataset is distributed under Community Data License Agreement – Permissive – Version 1.0
            https://cdla.io/permissive-1-0/
        """
    _dataset_name = 'iwildcam'
    _versions_dict = {
        '2.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/',
            'compressed_size': 11_957_420_032}}
    _annotations_versions_dict = {
        "1.0": {
            "download_url": 'https://worksheets.codalab.org/rest/bundles/0x9c36b6027a32425b89d6dbfe2dc7999d/contents/blob/',
            "compressed_size": 196_494_121}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')

        # path; download main dataset from codalab bundle
        # bboxes and segmentations masks are expected to be found in root_dir/, not root_dir/iwildcam_v_x
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        self._bbox_path, self._mask_dir = (Path(p) for p in self.initialize_masks_bboxes(root_dir, download))
        
        # Load splits
        df = pd.read_csv(self._data_dir / 'metadata.csv')

        # Splits
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        self._split_names = {'train': 'Train', 'val': 'Validation (OOD/Trans)',
                                'test': 'Test (OOD/Trans)', 'id_val': 'Validation (ID/Cis)',
                                'id_test': 'Test (ID/Cis)'}

        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._split_array = df['split_id'].values

        # Filenames
        self._input_array = df['filename'].values

        # Labels
        self._y_array = torch.tensor(df['y'].values)
        self._n_classes = max(df['y']) + 1
        self._y_size = 1
        assert len(np.unique(df['y'])) == self._n_classes

        # Location/group info
        n_groups = max(df['location_remapped']) + 1
        self._n_groups = n_groups
        assert len(np.unique(df['location_remapped'])) == self._n_groups

        # Sequence info
        n_sequences = max(df['sequence_remapped']) + 1
        self._n_sequences = n_sequences
        assert len(np.unique(df['sequence_remapped'])) == self._n_sequences

        # Extract datetime subcomponents and include in metadata
        df['datetime_obj'] = df['datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
        df['year'] = df['datetime_obj'].apply(lambda x: int(x.year))
        df['month'] = df['datetime_obj'].apply(lambda x: int(x.month))
        df['day'] = df['datetime_obj'].apply(lambda x: int(x.day))
        df['hour'] = df['datetime_obj'].apply(lambda x: int(x.hour))
        df['minute'] = df['datetime_obj'].apply(lambda x: int(x.minute))
        df['second'] = df['datetime_obj'].apply(lambda x: int(x.second))
        
        self.location_array = torch.tensor(df['location_remapped'].values)
        
        # cluster information
        cluster_array = np.zeros(len(self.y_array))
        for i in range(len(CLUSTERS)): 
            cluster_array[np.isin(self.location_array, CLUSTERS[i])] = i
        self.cluster_array = torch.tensor(cluster_array)

        # Big v. small animals, day v. night animals
        # 0 = label 0, 1 = night, 2 = day
        # 0 = label 0, 1 = small, 2 = big
        big_array = (np.isin(self.y_array, BIG_ANIMALS)).astype(int) + 1 
        big_array[self.y_array == 0] = 0
        day_array = (np.isin(self.y_array, DAY_ANIMALS)).astype(int) + 1 
        day_array[self.y_array == 0] = 0

        
        self._metadata_array = torch.tensor(np.stack([df['location_remapped'].values,
                            df['sequence_remapped'].values,
                            df['year'].values, df['month'].values, df['day'].values,
                            df['hour'].values, df['minute'].values, df['second'].values,
                            self.y_array, big_array, day_array, cluster_array], axis=1))
        self._metadata_fields = ['location', 'sequence', 'year', 'month', 'day', 'hour', 'minute', 'second', 'y', 'is_big', 'is_day', 'cluster']

        # additional info for the copy-paste augmentation
        self.img_ids = df['image_id']    
        self.bbox_df = pd.DataFrame(json.load(open(self._bbox_path, 'r'))['images']).set_index('id')    
        self.empty_indices = {
            'all': np.where(self._y_array == 0)[0],
            **{k: np.where(
                (self._y_array == 0) & (self._split_array == v)
            )[0] for k,v in self._split_dict.items()}
        }
        self.hour_array = self._metadata_array[:, self._metadata_fields.index('hour')]
        self.y_to_observed_locs = defaultdict(torch.Tensor, {y.item(): torch.unique(self.location_array[self._y_array == y]) for y in torch.unique(self._y_array)})

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(['location']))

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

        # All images are in the train folder
        img_path = self.data_dir / 'train' / self._input_array[idx]
        img = Image.open(img_path)

        return img

    def get_bbox(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - bboxes (list of lists): list of bboxes
                bboxes are [x, y, w, h] with (x,y) being the top left of the box
            - conf (float): megadetector max confidence over the bboxes
        """
        try:
            image_id = self.img_ids.loc[idx]
            bbox, conf = self.bbox_df.loc[image_id]['detections'], self.bbox_df.loc[image_id]['max_detection_conf']
            bbox = [b['bbox'] for b in bbox]
            return bbox, conf
        except:
            return [], 0.0

    def get_mask(self, idx, resize_wh=None, pad_wh=None):
        """
        Args:
            - idx (int): Index of a data point
            - resize_wh (int, int, ...): width/height to resize mask to
                this shouldn't be necessary for the iWildCam2021 masks (already padded to the same shape as the image)
            - pad_wh (int, int, ...): width/height to pad mask to
                this shouldn't be necessary for the iWildCam2021 masks (already padded to the same shape as the image)
        Output:
            - mask (Tensor): boolean segmentation mask of the idx-th data point 
        """
        try:
            image_id = self.img_ids.loc[idx]
            mask = Image.open(f'{self._mask_dir}/{image_id}.png')
            if resize_wh is not None: 
                mask = mask.resize(resize_wh[:2])
            mask = F.to_tensor(mask)
            if pad_wh is not None:
                w, h = mask.shape
                gap_w, gap_h = pad_wh[0] - w, pad_wh[1] - h
                F.pad(mask, (gap_w/2, gap_h/2, gap_w/2, gap_h/2), fill=0)
            mask = mask.squeeze().numpy() > 0
            return Image.fromarray(mask)
        except:
            return None

    def initialize_masks_bboxes(self, root_dir, download):
        mask_dir = os.path.join(root_dir, 'instance_masks')
        bbox_file = os.path.join(root_dir, 'megadetector_results.json')
        if not os.path.exists(mask_dir) or not os.path.exists(bbox_file):
            self.download_bundle(self._annotations_versions_dict, root_dir, download, version='1.0')        
        return bbox_file, mask_dir