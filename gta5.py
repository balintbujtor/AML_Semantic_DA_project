import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GTA5(Dataset):
    
    def __init__(self,
                 root_dir: str,
                 mode: str = 'train',
                 imgtransforms = None,):
        """
        Initializes the GTA5 dataset

        Args:
            root_dir (str): The root directory of the dataset
            mode (str, optional): mode of the usage: training or validation. Defaults to 'train'.
            imgtransforms (_type_, optional): image preprocessing transformations . Defaults to None.
        """
        super(GTA5, self).__init__()
        
        assert mode in ['train', 'val'], "mode should be 'train' or 'val'"
        
        self.mode = mode
        self.transforms = imgtransforms
        self.root_dir = root_dir
        
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            #TODO: Normalize
            ])
    

        image_dir = os.path.join(self.root_dir, 'images')
        label_dir = os.path.join(self.root_dir, 'labels')
        
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))
        
        self.image_paths = []
        self.label_paths = []
        
        self.image_paths.extend([os.path.join(image_dir, image) for image in image_files])
        self.label_paths.extend([os.path.join(label_dir, label) for label in label_files])
        
        assert len(self.image_paths) == len(self.label_paths), "Number of images and labels should be the same"
        print(f"Found {len(self.image_paths)} images and {len(self.label_paths)} labels")
        
        
    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the image and label at the given index

        Args:
            idx (int): The index of the image and label

        Returns:
            tuple: The image and label
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transforms is not None and self.mode == 'train':
            image = self.transforms(image)
        image = self.to_tensor(image)
        
        label = Image.open(self.label_paths[idx])
        #TODO: for some reason there are labels with values that are greater than 19
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return (image, label)
    
    
    def __len__(self) -> int:
        """
        Returns the number of images in the dataset

        Returns:
            int:
        """
        return len(self.image_paths)