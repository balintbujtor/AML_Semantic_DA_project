from torchvision.transforms import v2
import torch

# Define the augmentations
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
aug_transformations = {
    "C-S-HF": v2.Compose([contrast, saturation, rdHflip]),
    "H-RP-HF": v2.Compose([hue, rdPerspective, rdHflip]),
    "B-GS-R": v2.Compose([brightness, grayscale, rdRotation]),
    "S-BL-R": v2.Compose([rdSolarize, blur, rdRotation])
}

label_transformations = {
    "C-S-HF": v2.Compose([rdHflip]),
    "H-RP-HF": v2.Compose([rdHflip]),
    "B-GS-R": v2.Compose([rdRotation]),
    "S-BL-R": v2.Compose([rdRotation])
}
