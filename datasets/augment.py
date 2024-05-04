from torchvision.transforms import v2

bright_t = v2.ColorJitter(brightness = [1,3])
contrast_t = v2.ColorJitter(contrast = [2,6])
saturation_t = v2.ColorJitter(saturation = [1,4])
hue_t = v2.ColorJitter(hue = 0.3)
gs_t = v2.Grayscale(3)
hflip_t = v2.RandomHorizontalFlip(p = 1)
rp_t = v2.RandomPerspective(p = 1, distortion_scale = 0.5)
rot_t = v2.RandomRotation(degrees = 90)
blur_t = v2.GaussianBlur(kernel_size=15, sigma=(0.3, 0.7))
sol_t = v2.RandomSolarize(p = 1, threshold = 0.4)


aug_transformations = {
    "C-S-HF": v2.Compose([contrast_t, saturation_t, hflip_t]),
    "H-RP-HF": v2.Compose([hue_t, rp_t, hflip_t]),
    "B-GS-R": v2.Compose([bright_t, gs_t, rot_t]),
    "S-BL-R": v2.Compose([sol_t, blur_t, rot_t])
}

label_transformations = {
    "C-S-HF": v2.Compose([hflip_t]),
    "H-RP-HF": v2.Compose([hflip_t]),
    "B-GS-R": v2.Compose([rot_t]),
    "S-BL-R": v2.Compose([rot_t])
}