#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os
from PIL import Image
from torch.utils.data import Dataset


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class CityScapes(Dataset):
    """
    A class to represent the CityScapes dataset.
    """
    
    def __init__(self, root_dir: str, mode: str = 'train', transforms=None, cities: list = []):
        """
        Initialize the CityScapes dataset.

        Parameters:
        root_dir (str): The root directory of the dataset.
        mode (str, optional): The mode of the dataset, either 'train' or 'val'. Default is 'train'.
        transforms (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. Default is None.
        cities (list of str, optional): The cities to include in the dataset. If None, all cities are included. Default is None.

        Attributes:
        mode (str): The mode of the dataset.
        root_dir (str): The root directory of the dataset.
        cities (list of str): The cities included in the dataset.
        image_dir (str): The directory of the images.
        label_dir (str): The directory of the labels.
        image_paths (list of str): The paths of the image files.
        label_paths (list of str): The paths of the label files.

        Raises:
        AssertionError: If mode is not 'train' or 'val'.
        AssertionError: If cities is not a list.
        AssertionError: If any element in cities is not a string.
        AssertionError: If the number of images and labels is not the same.
        """
        super(CityScapes, self).__init__()

        assert mode in ['train', 'val'], "mode should be 'train' or 'val'"
        
        self.mode = mode
        self.transforms = transforms
        self.root_dir = root_dir
        
        if cities is not None:
            assert isinstance(cities, list), "cities should be a list"
            assert all(isinstance(city, str) for city in cities), "all elements in cities should be a string"
            self.cities = cities
        else:
            self.cities = sorted(os.listdir(os.path.join(self.root_dir, 'Cityspaces', 'images', self.mode)))

        self.image_dir = os.path.join(self.root_dir, 'Cityspaces', 'images', self.mode)
        self.label_dir = os.path.join(self.root_dir, 'Cityspaces', 'gtFine', self.mode)

        self.image_paths = []
        self.label_paths = []
        self.colour_map_path = []
        
        for city in self.cities:
            city_image_dir = os.path.join(self.image_dir, city)
            city_label_dir = os.path.join(self.label_dir, city)

            city_image_files = sorted(os.listdir(city_image_dir))
            city_label_files = sorted(os.listdir(city_label_dir))

            self.image_paths.extend([os.path.join(city_image_dir, file) for file in city_image_files])
            self.label_paths.extend([os.path.join(city_label_dir, file) for file in city_label_files if 'labelTrainIds' in file])
            self.colour_map_path.extend([os.path.join(city_label_dir, file) for file in city_label_files if 'color' in file])

        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels should be the same"
        print(f"Found {len(self.image_paths)} {self.mode} images")
    
    def __getitem__(self, idx: int) -> tuple:
        """Returns the image and label at the given index

        Args:
            idx (int): The index of the image and label

        Returns:
            tuple: The image and label
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = pil_loader(image_path)
        
        # apply the transformations, if any
        if self.transforms is not None:
            image = self.transforms(image)
        
        label = pil_loader(label_path)
        
        return image, label

    def __len__(self) -> int:
        """returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.image_paths)
