import os
import random
import cv2
import numpy as np

from torch.utils.data import Dataset
from .util import *


class Dataset(Dataset):
    """
    Dataset that returns:
      - 'jpg': the main image in [-1,1] float32
      - 'txt': the text prompt
      - 'local_conditions': an array with shape [H, W, 3*N], in [0,1] float range
    """
    def __init__(self,
                 anno_path,
                 image_dir,
                 condition_root,
                 local_type_list,
                 global_type_list,
                 resolution,
                 drop_txt_prob,
                 keep_all_cond_prob,
                 drop_all_cond_prob,
                 drop_each_cond_prob):
        
        # Read .txt file or CSV with image IDs and captions
        file_ids, self.annos = read_anno(anno_path)

        # Paths to full-resolution images
        self.image_paths = [os.path.join(image_dir, file_id + '.jpg') for file_id in file_ids]

        # Paths to local condition images (e.g. canny, hed, midas)
        self.local_paths = {}
        for local_type in local_type_list:
            self.local_paths[local_type] = [
                os.path.join(condition_root, local_type, file_id + '.jpg') for file_id in file_ids
            ]

        # Because we do not use global conditions at all, skip them
        # (global_type_list is ignored)

        self.local_type_list = local_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        # 1) Main image
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))
        image = (image.astype(np.float32) / 127.5) - 1.0  # -> [-1,1]

        # 2) Caption
        anno = self.annos[index]

        # 3) Local condition images
        local_files = []
        for local_type in self.local_type_list:
            local_files.append(self.local_paths[local_type][index])

        local_conditions = []
        for local_file in local_files:
            cond_img = cv2.imread(local_file)
            cond_img = cv2.cvtColor(cond_img, cv2.COLOR_BGR2RGB)
            cond_img = cv2.resize(cond_img, (self.resolution, self.resolution))
            cond_img = cond_img.astype(np.float32) / 255.0  # -> [0,1]
            local_conditions.append(cond_img)

        # Possibly drop the text prompt
        if random.random() < self.drop_txt_prob:
            anno = ''

        # Possibly keep/drop local conditions
        local_conditions = keep_and_drop(
            local_conditions,
            self.keep_all_cond_prob,
            self.drop_all_cond_prob,
            self.drop_each_cond_prob
        )

        # If there are no conditions left after dropping, we put a dummy array
        if len(local_conditions) != 0:
            # Concatenate along channel dimension: shape [H, W, 3*N]
            local_conditions = np.concatenate(local_conditions, axis=2)
        else:
            # shape [1,1,1] or something minimal
            local_conditions = np.zeros((1,1,1), dtype=np.float32)

        return dict(
            jpg=image,
            txt=anno,
            local_conditions=local_conditions
        )
