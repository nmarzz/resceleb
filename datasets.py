import torch
import numpy as np
from torchvision import datasets, transforms

from functools import partial
import os
import PIL
from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets import VisionDataset
import yaml

config = open('config.yaml', 'r')
parsed_config = yaml.load(config, Loader=yaml.FullLoader)

# Get values from index
attributes = parsed_config['attributes']
hair_color_vals = parsed_config['hair_color_vals']
hair_color_idxs = [i for i,val in enumerate(attributes) if val in hair_color_vals]
hair_style_vals = parsed_config['hair_style_vals']
hair_style_idxs = [i for i,val in enumerate(attributes) if val in hair_style_vals]




class CelebA(VisionDataset):
    """`Large-scale CelebFaces Attributes (CelebA) Dataset <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>`_ Dataset.

    Adapted from the original Pytorch code here:
    https://pytorch.org/vision/0.8/_modules/torchvision/datasets/celeba.html
    
    """

    base_folder = "celeba"

    def __init__(
            self,
            root: str,
            mode : str,
            split: str = "train",
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download = False            
    ) -> None:
        import pandas
        super(CelebA, self).__init__(root, transform=transform,target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError('target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "val": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[split]

        fn = partial(os.path.join, self.root, self.base_folder)        
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)
        bbox = pandas.read_csv(fn("list_bbox_celeba.txt"), delim_whitespace=True, header=1, index_col=0)
        landmarks_align = pandas.read_csv(fn("list_landmarks_align_celeba.txt"), delim_whitespace=True, header=1)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)
        self.mode = mode


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:            
            target = tuple(target) if len(target) > 1 else target[0]
            
            if self.mode == 'hair_style':
                target = target[hair_style_idxs]                                    
                target = torch.argmax(target) if torch.sum(target) != 0 else torch.tensor(len(hair_style_idxs))           
            elif self.mode == 'hair_color':
                target = target[hair_color_idxs]                                    
                target = torch.argmax(target) if torch.sum(target) != 0 else torch.tensor(len(hair_color_idxs))                   

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

