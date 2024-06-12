from torchvision.transforms import v2
import torch
from PIL import Image

CITYSCAPES_CROP = (512, 1024)
GTA5_CROP = (526,957)

# Imagenet mean and std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
    
# Define specific augmentations
brightness = v2.ColorJitter(brightness = [1,3])
contrast = v2.ColorJitter(contrast = [2,6])
saturation = v2.ColorJitter(saturation = [1,4])
hue = v2.ColorJitter(hue = 0.3)
grayscale = v2.Grayscale(3)
rdHflip = v2.RandomHorizontalFlip(p = 1)
rdPerspective = v2.RandomPerspective(p = 1, distortion_scale = 0.5)
rdRotation = v2.RandomRotation(degrees = 90)
blur = v2.GaussianBlur(kernel_size=15, sigma=(0.3, 0.7))
rdSolarize = v2.RandomSolarize(p = 1, threshold = 0.4)

to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])


# Define standard transformations
img_std_transformations = {
    # this can be source of error float16 should be the datatype (before it was float32)
    "std_cityscapes" : v2.Compose([v2.Resize((512, 1024), Image.BILINEAR),v2.ToImage(), v2.ToDtype(torch.float16, scale=True),v2.Normalize(mean=MEAN, std=STD)]),
    "std_gta5" : v2.Compose([v2.Resize((526,957), Image.BILINEAR),v2.ToImage(), v2.ToDtype(torch.float16, scale=True),v2.Normalize(mean=MEAN, std=STD)])
}

lbl_std_transformations = {
    # Image.NEAREST s.t. the label values are kept
    # PILToTensor() to avoid normalizing into (0,1),
    "std_cityscapes" : v2.Compose([v2.Resize((512, 1024),Image.NEAREST),v2.PILToTensor()]), 
    "std_gta5" : v2.Compose([v2.Resize((526,957),Image.NEAREST),v2.PILToTensor()])
}
    

# Define series of augmentations
img_aug_transformations = {
    "C-S-HF": v2.Compose([contrast, saturation, rdHflip]),
    "H-RP-HF": v2.Compose([hue, rdPerspective, rdHflip]),
    "B-GS-R": v2.Compose([brightness, grayscale, rdRotation]),
    "S-BL-R": v2.Compose([rdSolarize, blur, rdRotation])
}

lbl_aug_transformations = {
    "C-S-HF": v2.Compose([rdHflip]),
    "H-RP-HF": v2.Compose([rdHflip]),
    "B-GS-R": v2.Compose([rdRotation]),
    "S-BL-R": v2.Compose([rdRotation])
}
