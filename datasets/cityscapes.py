#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
import numpy as np
from collections import namedtuple
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from typing import Any, Tuple
import random as rd
import utils.transforms as transforms

class CityScapes(VisionDataset):
    """
    A class to represent the CityScapes dataset.
    """
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("unlabeled",            0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle",          1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi",           3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static",               4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic",              5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground",               6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road",                 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk",             8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking",              9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track",           10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building",             11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall",                 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence",                13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail",           14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge",               15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel",               16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole",                 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup",            18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light",        19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign",         20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation",           21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain",              22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky",                  23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person",               24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider",                25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car",                  26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck",                27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus",                  28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan",              29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer",              30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train",                31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle",           32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle",              33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate",        -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    id_to_color = np.array([c.color for c in classes])
    
    
    def __init__(self,
                 aug_method,
                 split: str = 'train',
                 is_pseudo: bool = False, 
        ):
        """
        Initializes the CityScapes dataset.
        1. Initializes the variables and ensures the correctness of the split
        2. Cycles through the cities to save the image and label paths and ensures their lengths match
        
        Args:
            aug_method (str): the augmentation method to use. Empty string for no aug, string code for the given aug method.
            split (str, optional): train test or validation split. Defaults to 'train'.
            is_pseudo (bool, optional): whether to use pseudo labels or not. Defaults to False.
        """
        
        super(CityScapes, self).__init__()

        assert split in ['train', 'val'], "split should be 'train', or 'val'"
        
        self.root_dir = "Cityscapes/Cityspaces/"    
        self.split = split
        self.aug_method = aug_method

        cities = sorted(os.listdir(os.path.join(self.root_dir, 'images', self.split)))

        label_dir = 'pseudo_labels' if is_pseudo else 'gtFine'
        
        self.image_dir = os.path.join(self.root_dir, 'images', self.split)
        self.label_dir = os.path.join(self.root_dir, label_dir, self.split)

        self.image_paths = []
        self.label_paths = []
        self.colour_map_path = []
        
        for city in cities:
            city_image_dir = os.path.join(self.image_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)

            city_image_files = sorted(os.listdir(city_image_dir))
            city_label_files = sorted(os.listdir(city_label_dir))

            self.image_paths.extend([os.path.join(city_image_dir, file) for file in city_image_files])
            self.label_paths.extend([os.path.join(city_label_dir, file) for file in city_label_files if 'labelTrainIds' in file])
            self.colour_map_path.extend([os.path.join(city_label_dir, file) for file in city_label_files if 'color' in file])

        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels should be the same"
        print(f"Found {len(self.image_paths)} images for {self.split}")

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the image and label at the given index
        Performs the specified augmentation methods with a given probability
        
        Args:
            idx (int): The index of the image and label

        Returns:
            tuple: The image and label
        """
        img = Image.open(self.image_paths[idx]).convert('RGB')
        lbl = Image.open(self.label_paths[idx])

        # Apply std transformation
        if self.aug_method == '':
            img = transforms.img_std_transformations["std_cityscapes"](img)
            lbl = transforms.lbl_std_transformations["std_cityscapes"](lbl)

        # no normalization for fda
        elif self.aug_method == 'nonorm':
            img = transforms.img_nonorm_transformations["std_cityscapes"](img)
            lbl = transforms.lbl_nonorm_transformations["std_cityscapes"](lbl)
            
        # Apply augmentation
        elif self.aug_method != '' and self.aug_method != 'nonorm':
            if rd.random() < 0.5:
                img = transforms.img_aug_transformations[self.aug_method](img)
                lbl = transforms.lbl_aug_transformations[self.aug_method](lbl)
        
        
        return img, lbl

    def __len__(self) -> int:

        return len(self.image_paths)
    
    
    @classmethod
    def encode_target(cls, target):
        """Encodes the target to the train_id

        Args:
            target (int): the target to encode

        Returns:
            int: encoded target, aka train_id
        """
        return cls.id_to_train_id[np.array(target)]


    @classmethod
    def decode_target(cls, target):
        """Decodes the target from train_id to color

        Args:
            target (int): the train_id to decode

        Returns:
            _type_: decoded train_id to color
        """
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]


    @classmethod 
    def visualize_prediction(cls,outputs,labels) -> Tuple[Any, Any]:
        """Visualizes the predictions

        Args:
            outputs (_type_): the image that the net generated
            labels (_type_): the corresponding, correct label

        Returns:
            Tuple[Any, Any]: The colorized predictions and the colorized labels
        """
        preds = outputs.max(1)[1].detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()
        colorized_preds = cls.decode_target(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
        colorized_labels = cls.decode_target(lab).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
        colorized_labels = Image.fromarray(colorized_labels[0])
        return colorized_preds , colorized_labels
   