from datetime import datetime
from pathlib import Path
import os

from PIL import Image
import pandas as pd
import numpy as np
import torch
import json
from torchvision.transforms import functional as F
from collections import defaultdict

from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSUnlabeledDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy, Recall, F1

# sample of 500 bboxes has mean height >= 20% of the frame or not
BIG_ANIMALS = [1, 4, 5, 6, 7, 8, 10, 12, 13, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 32, 34, 35, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 52, 53, 54, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 78, 81, 82, 87, 88, 89, 90, 93, 94, 96, 97, 99, 100, 101, 102, 103, 104, 106, 107, 109, 111, 112, 113, 114, 115, 116, 118, 119, 125, 126, 127, 133, 134, 135, 142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 157, 158, 159, 167, 168, 172, 174, 175, 179, 180]
SMALL_ANIMALS = [128, 129, 2, 3, 130, 131, 132, 136, 9, 137, 11, 138, 139, 14, 140, 141, 17, 18, 148, 27, 156, 31, 160, 33, 161, 162, 36, 163, 38, 164, 165, 166, 169, 43, 170, 171, 46, 173, 176, 177, 50, 51, 178, 181, 55, 61, 75, 76, 79, 80, 83, 84, 85, 86, 91, 92, 95, 98, 105, 108, 110, 117, 120, 121, 122, 123, 124]

# mean hour is w/n working hours 9a-5p v. not
DAY_ANIMALS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44, 46, 48, 49, 51, 52, 53, 54, 56, 58, 59, 60, 61, 64, 65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 89, 90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 123, 126, 127, 130, 132, 133, 135, 136, 137, 138, 139, 140, 141, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 157, 159, 161, 162, 163, 164, 165, 166, 168, 170, 171, 172, 173, 174, 176, 177, 181]
NIGHT_ANIMALS = [128, 129, 131, 134, 9, 11, 142, 143, 17, 19, 22, 154, 156, 29, 158, 160, 38, 167, 169, 45, 47, 175, 50, 178, 179, 180, 55, 57, 62, 63, 66, 67, 79, 83, 88, 91, 92, 94, 110, 122, 124, 125]


class IWildCamUnlabeledDataset(WILDSUnlabeledDataset):
    """
    The unlabeled iWildCam2020-WILDS dataset.
    This is a modified version of the original iWildCam2020 competition dataset.
    Input (x):
        RGB images from camera traps
    Metadata:
        Each image is annotated with the ID of the location (camera trap) it came from.
    Website:
        http://lila.science/datasets/wcscameratraps
        https://library.wcs.org/ScienceData/Camera-Trap-Data-Summary.aspx
    Original publication:
        @misc{wcsdataset,
          title = {Wildlife Conservation Society Camera Traps Dataset},
          howpublished = {\\url{http://lila.science/datasets/wcscameratraps}},
        }
    License:
        This dataset is distributed under Community Data License Agreement – Permissive – Version 1.0
        https://cdla.io/permissive-1-0/
    """

    _dataset_name = "iwildcam_unlabeled"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xff56ea50fbf64aabbc4d09b2e8d50e18/contents/blob/",
            "compressed_size": 41_016_937_676,
        }
    }
    _annotations_versions_dict = {
        "1.0": {
            "download_url": 'https://worksheets.codalab.org/rest/bundles/0x9c36b6027a32425b89d6dbfe2dc7999d/contents/blob/',
            "compressed_size": 196_494_121}}

    def __init__(
        self, version=None, root_dir="data", download=False, split_scheme="official"
    ):

        self._version = version
        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")

        # path
        # bboxes and segmentations masks are expected to be found in root_dir/, not root_dir/iwildcam_v_x
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        self._bbox_path, self._mask_dir = (Path(p) for p in self.initialize_masks_bboxes(root_dir, download))

        # Load splits
        df = pd.read_csv(self._data_dir / "metadata.csv")

        # Splits
        self._split_dict = {"extra_unlabeled": 0}
        self._split_names = {"extra_unlabeled": "Extra Unlabeled"}
        df["split_id"] = 0
        self._split_array = df["split_id"].values

        # Filenames
        df["filename"] = df["uid"].apply(lambda x: x + ".jpg")
        self._input_array = df["filename"].values

        # Location/group info
        n_groups = df["location_remapped"].nunique()
        self._n_groups = n_groups

        def get_date(x):
            if isinstance(x, str):
                return datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
            else:
                return -1

        ## Extract datetime subcomponents and include in metadata
        df["datetime_obj"] = df["datetime"].apply(get_date)
        df["year"] = df["datetime_obj"].apply(
            lambda x: int(x.year) if isinstance(x, datetime) else -1
        )
        df["month"] = df["datetime_obj"].apply(
            lambda x: int(x.month) if isinstance(x, datetime) else -1
        )
        df["day"] = df["datetime_obj"].apply(
            lambda x: int(x.day) if isinstance(x, datetime) else -1
        )
        df["hour"] = df["datetime_obj"].apply(
            lambda x: int(x.hour) if isinstance(x, datetime) else -1
        )
        df["minute"] = df["datetime_obj"].apply(
            lambda x: int(x.minute) if isinstance(x, datetime) else -1
        )
        df["second"] = df["datetime_obj"].apply(
            lambda x: int(x.second) if isinstance(x, datetime) else -1
        )

        df["y"] = df["y"].apply( # filter out "bad" labels (-1 means the category was not in iwildcam_v2.0; 99999 means the category was unknown). map all to -100.
            lambda x: x if ((x != -1) and (x != 99999)) else -100
        )
        self._y_array = torch.LongTensor(df['y'].values)

        # we have no known geocoordinates
        cluster_array = np.zeros(len(self._y_array))

        # Big v. small animals, day v. night animals
        # 0 = label 0, 1 = night, 2 = day
        # 0 = label 0, 1 = small, 2 = big
        big_array = (np.isin(self._y_array, BIG_ANIMALS)).astype(int) + 1 
        big_array[self._y_array <= 0] = 0
        day_array = (np.isin(self._y_array, DAY_ANIMALS)).astype(int) + 1 
        day_array[self._y_array <= 0] = 0

        self._metadata_array = torch.tensor(
            np.stack(
                [
                    df["location_remapped"].values,
                    df["sequence_remapped"].values,
                    df["year"].values,
                    df["month"].values,
                    df["day"].values,
                    df["hour"].values,
                    df["minute"].values,
                    df["second"].values,
                    df["y"],
                    big_array,
                    day_array,
                    cluster_array
                ],
                axis=1,
            )
        )
        self._metadata_fields = [
            "location",
            "sequence",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "y",
            "is_big",
            "is_day",
            "cluster",
        ]

        # Bboxes and segmentation masks
        # Files expected to be found in root_dir/
        self.img_ids = df['uid']
        self.bbox_df = pd.DataFrame(json.load(open(f"{root_dir}/megadetector_results.json", 'r'))['images']).set_index('id')
        self.mask_dir = Path(f"{root_dir}/instance_masks")
        
        # additional info for the copy-paste augmentation
        self.img_ids = df['uid']    
        self.bbox_df = pd.DataFrame(json.load(open(self._bbox_path, 'r'))['images']).set_index('id')    
        self.empty_indices = {
            'all': np.where(self._y_array == 0)[0],
            **{k: np.where(
                (self._y_array == 0) & (self._split_array == v)
            )[0] for k,v in self._split_dict.items()}
        }
        self.location_array = torch.tensor(df["location_remapped"].values)
        self.hour_array = self._metadata_array[:, self._metadata_fields.index('hour')]
        self.cluster_array = torch.tensor(cluster_array)
        self.y_to_observed_locs = defaultdict(torch.Tensor, {y.item(): torch.unique(self.location_array[self._y_array == y]) for y in torch.unique(self._y_array)})

        # eval grouper
        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=(["location"])
        )

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (Tensor): Input features of the idx-th data point
        """

        # All images are in the train folder
        img_path = self.data_dir / "images" / self._input_array[idx]
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
            mask = mask.squeeze().numpy()  > 0
            return Image.fromarray(mask)
        except:
            return None

    def initialize_masks_bboxes(self, root_dir, download):
        mask_dir = os.path.join(root_dir, 'instance_masks')
        bbox_file = os.path.join(root_dir, 'megadetector_results.json')
        if not os.path.exists(mask_dir) or not os.path.exists(bbox_file):
            self.download_bundle(self._annotations_versions_dict, root_dir, download, version='1.0')        
        return bbox_file, mask_dir