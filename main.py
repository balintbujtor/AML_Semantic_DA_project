from numpy import False_
from torchvision import transforms
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from torch.utils.data import DataLoader
from utils import *
from train import train, val
import os

def main():
    args = parse_args()

    n_classes = args.num_classes
    root_dir = args.root_dir
    split = args.split
    chosen_dataset = args.dataset

    validation_split = .2
    shuffle_dataset = True
    random_seed = 42

    np.random.seed(random_seed)
    os.environ['SEED'] = str(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    CITYSCAPES_CROP = (512, 1024)
    GTA5_CROP = (526,957)
    
    # Imagenet mean and std
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    assert chosen_dataset in ['cityscapes', 'gta5'], "Dataset not supported"
    
    if chosen_dataset == 'cityscapes':
        
        std_img_transforms = transforms.Compose([
            transforms.Resize((CITYSCAPES_CROP), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        
        # Image.NEAREST s.t. the label values are kept
        # PILToTensor() to avoid normalizing into (0,1)
        std_lbl_transforms = transforms.Compose([
            transforms.Resize((CITYSCAPES_CROP),Image.NEAREST),
            transforms.PILToTensor(),
        ])
        
        train_dataset = CityScapes(root_dir=root_dir, split=split, img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms)
        
        dataloader_train = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)

        val_dataset = CityScapes(root_dir=root_dir, split='val', img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms)
        dataloader_val = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)
        
    elif chosen_dataset == 'gta5':
        
        std_img_transforms = transforms.Compose([
            transforms.Resize((GTA5_CROP), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
        
        std_lbl_transforms = transforms.Compose([
            transforms.Resize((GTA5_CROP), Image.NEAREST),
            transforms.PILToTensor(),
        ])
        
        dataset = GTA5(root=Path(args.root_dir), img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms)
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        
        if shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        dataloader_train = DataLoader(dataset, 
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      pin_memory=False,
                                      drop_last=True, 
                                      sampler=train_sampler)
        
        dataloader_val = DataLoader(dataset, 
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False,
                                    sampler=valid_sampler)


    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=n_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
        
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
        
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    ## train loop
    train(args, model, optimizer, dataloader_train, dataloader_val, device)
    # final test
    val(args, model, dataloader_val, device)

if __name__ == "__main__":
    main()
        