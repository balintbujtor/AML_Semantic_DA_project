import os
from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple
import numpy as np
from PIL import Image
from torch.utils.data import Dataset as torchDataset
from datasets.cityscapes import CityScapes
import random as rd
import utils.transforms as transforms
from pathlib import Path

class BaseGTALabels(metaclass=ABCMeta):
    pass


@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
    ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret




class GTA5(torchDataset):
    """
    This class handles the GTA5 dataset.
    It contains the PathPair_ImgAndLabel class that is used to create the list of image and label paths.
    """
    label_map = GTA5Labels_TaskCV2017()

    train_id = np.array([label.train_id for label in CityScapes.classes])
    
    class PathPair_ImgAndLabel:
        IMG_DIR_NAME = "images"
        LBL_DIR_NAME = "labels"
        SUFFIX = ".png"

        def __init__(self, root, labels_source: str = "GTA5"):
            self.root = root
            self.img_paths = self.create_imgpath_list()
            self.labels_source = labels_source
            self.lbl_paths = self.create_lblpath_list()

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int):
            img_path = self.img_paths[idx]
            lbl_path = self.lbl_paths[idx]
            return img_path, lbl_path

        def create_imgpath_list(self):
            img_dir = Path(os.path.join(self.root,self.IMG_DIR_NAME))
            img_path = [path for path in img_dir.glob(f"*{self.SUFFIX}")]
            return img_path

        def create_lblpath_list(self):
            lbl_dir = os.path.join(self.root,self.LBL_DIR_NAME)
            if self.labels_source == "cityscapes":
                lbl_path = [os.path.join(lbl_dir,path) for path in os.listdir(lbl_dir) if path.endswith(self.SUFFIX) and path.__contains__("_labelTrainIds")]
            elif self.labels_source == "GTA5":
                lbl_path = [os.path.join(lbl_dir,path) for path in os.listdir(lbl_dir) if (path.endswith(self.SUFFIX) and not path.__contains__("_labelTrainIds.png"))]
            return lbl_path

    def __init__(self, 
                 aug_method: str,
                 training_method: str = '',
                ):
        """
        Initializes the variables and saves the augmentation method.

        Args:
            aug_method (str): the augmentation method to use. Empty string for no aug, string code for the given aug method.
            training_method (str, optional): If we use FDA training the image size of GTA5 and Cityscapes have to match.
                In this case a different img transformation is used that makes the sizes equal. 
                Defaults to ''.
        """
        self.root = "GTA5"
        self.training_method = training_method
        self.labels_source = self.root                   
        self.paths = self.PathPair_ImgAndLabel(root=self.root, labels_source=self.labels_source)
        self.aug_method = aug_method
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int, isPath: bool = False):
        """
            Returns the item based on the idx parameter. 
            If isPath is True, it returns the paths of the image and label.
            If the training method is FDA, the image is resized to match the Cityscapes dataset.
            If an augmentation method is specified, it is applied with a probability of 0.5.
        Args:
            idx (int): _description_
            isPath (bool, optional): if true only returns the paths. Defaults to False.

        Returns:
            _type_: either the paths or the images and labels
        """
        img_path, lbl_path = self.paths[idx]
        if isPath:
            return img_path, lbl_path

        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)

        
        # IMPORTANT to convert to trainId to match with the Cityscapes labels
        lbl = Image.fromarray(np.array(self.convert_to_trainId(lbl),dtype='uint8'))

        # Apply std transformations
        if self.training_method == 'train_fda' or self.training_method == 'train_ssl_fda' :
            img = transforms.img_std_transformations["std_cityscapes"](img)
            lbl = transforms.lbl_std_transformations["std_cityscapes"](lbl)

        else:
            img = transforms.img_std_transformations["std_gta5"](img)
            lbl = transforms.lbl_std_transformations["std_gta5"](lbl)

        # Apply augmentation
        if self.aug_method != '':
            if rd.random() < 0.5:
                img = transforms.aug_transformations[self.aug_method](img)
                lbl = transforms.label_transformations[self.aug_method](lbl)
                      
        return img, lbl

    @staticmethod
    def read_img(path):
        img = Image.open(str(path))
        return img

    @classmethod
    def decode(cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)

    @staticmethod
    def _decode(lbl, label_map):
        # remap_lbl = lbl[np.where(np.isin(lbl, cls.label_map.support_id_list), lbl, 0)]
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl


    def convert_to_trainId(self, lbl):
        return self.train_id[np.array(lbl)]


    @classmethod 
    def visualize_prediction(cls,outputs,labels) -> tuple:
        """
        Args:
                cls (GTA5): The class object
                outputs (Tensor): The output of the model
                labels (Tensor): The ground truth labels
        Returns:
                Tuple[Any, Any]: The colorized predictions and the colorized labels
        """
        preds = outputs.max(1)[1].detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()
        colorized_preds = cls.decode(preds).astype('uint8') # To RGB images, (N, H, W, 3), ranged 0~255, numpy array
        colorized_labels = cls.decode(lab).astype('uint8')
        colorized_preds = Image.fromarray(colorized_preds[0]) # to PIL Image
        colorized_labels = Image.fromarray(colorized_labels[0])
        return colorized_preds , colorized_labels