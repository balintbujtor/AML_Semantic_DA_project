from torchvision.transforms import v2
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from torch.utils.data import DataLoader
from utils import *
import os
import trainings.train_1 as train_1
import trainings.train_ADA as train_ADA
import trainings.train_FDA as train_FDA
import datasets.augment as augment
 
def calculate_mean_std(dataloader):
    """Calculates the mean and std of the dataset

    Args:
        dataset (CityScapes): the dataset to calculate the mean and std of

    Returns:
        tuple: the mean and std of the dataset
    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def main():
    args = parse_args()

    root_dir = args.root_dir #currently not used, to reimplement to match new structure
    split = args.split

    val_dataset = args.validation_dataset if args.validation_dataset != '' else args.training_dataset
    val_only = True if args.validation_only else False

    aug_method = args.aug_method

    validation_split = .2
    shuffle_dataset = True

    # Fixing the random seeds
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
    
    assert args.training_dataset in ['','cityscapes', 'gta5'], "Dataset not supported"
    assert val_dataset in ['','cityscapes', 'gta5'], "Dataset not supported"
    
    # TODO: REMOVE AFTER COMPUTING CS MEAN AND STD
    img_transforms = v2.Compose([
        v2.ToTensor(),
    ])
    dataset = CityScapes(root_dir="Cityscapes/Cityspaces/", split='train', img_transforms=img_transforms, lbl_transforms=img_transforms)
    dataloader_train = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=True)
    
    mean, std = calculate_mean_std(dataloader_train)
    
    print(f"Mean: {mean}, Std: {std}")
    print('Entering while because mean and std are computed')

    while True:
        pass   
    
     
    #Loads cityscapes if it's used in train or val
    if 'cityscapes' in (args.training_dataset, val_dataset, args.target_dataset):
        print("Cityscapes loaded.")
        root_dir="Cityscapes/Cityspaces/"
        
        std_img_transforms = v2.Compose([
            v2.Resize((CITYSCAPES_CROP), Image.BILINEAR),
            v2.ToTensor(),
            v2.Normalize(mean=MEAN, std=STD),
        ])
        
        # Image.NEAREST s.t. the label values are kept
        # PILToTensor() to avoid normalizing into (0,1)
        std_lbl_transforms = v2.Compose([
            v2.Resize((CITYSCAPES_CROP),Image.NEAREST),
            v2.PILToTensor(),
        ])
        


        if args.training_dataset == 'cityscapes':
            print("dataloader_train is on cityscapes")
            train_dataset = CityScapes(root_dir=root_dir, split=split, img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms)
            dataloader_train = DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=False,
                                        drop_last=True)
            
        if args.target_dataset == 'cityscapes':
            print("dataloader_target is on cityscapes")
            target_dataset = CityScapes(root_dir=root_dir, split=split, img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms)
            dataloader_target = DataLoader(target_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=False,
                                        drop_last=True)

        if val_dataset == 'cityscapes':
            print("dataloader_val is on cityscapes")
            val_dataset = CityScapes(root_dir=root_dir, split='val', img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms)
            dataloader_val = DataLoader(val_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        drop_last=False)
            
    #Loads gta5 if it's used in train or val        
    if 'gta5' in (args.training_dataset,val_dataset, args.target_dataset):
        print("Gta5 loaded.")
        root_dir="GTA5"
        
        # if FDA is used the sizes of the images have to be the same
        if args.training_method == 'train_FDA': 
            GTA5_CROP = CITYSCAPES_CROP
        
        std_img_transforms = v2.Compose([
            v2.Resize((GTA5_CROP), Image.BILINEAR),
            augment.to_tensor,
            v2.Normalize(mean=MEAN, std=STD),
        ])
        
        std_lbl_transforms = v2.Compose([
            v2.Resize((GTA5_CROP), Image.NEAREST),
            v2.PILToTensor(),
        ])

                                                                         
        
        dataset = GTA5(root=Path(args.root_dir), img_transforms=std_img_transforms, lbl_transforms=std_lbl_transforms,aug_method=aug_method)
        #TODO Separate the gta5 class calls for each mode: train,...
        #TODO Implement aug for cityscapes
        #TODO Rename stuff augment.py
        
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        
        if shuffle_dataset :
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Setting the dataloaders
        if args.training_dataset == 'gta5':
            print("dataloader_train is on gta5")
            dataloader_train = DataLoader(dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        pin_memory=False,
                                        drop_last=True, 
                                        sampler=train_sampler)
            
        if args.target_dataset == 'gta5':
            print("dataloader_target is on gta5")
            dataloader_target = DataLoader(dataset, 
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        pin_memory=False,
                                        drop_last=True, 
                                        sampler=train_sampler)

        if val_dataset == 'gta5':
            print("dataloader_val is on gta5")
            dataloader_val = DataLoader(dataset, 
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        drop_last=False,
                                        sampler=valid_sampler)


    ## model
    model = BiSeNet(backbone=args.backbone, n_classes=args.num_classes, pretrain_model=args.pretrain_path, use_conv_last=args.use_conv_last)

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
        print('Optimizer not supported \n')
        return None

# build disc_optimizer
    if args.disc_optimizer == 'rmsprop':
        disc_optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
        
    elif args.disc_optimizer == 'sgd':
        disc_optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
        
    elif args.disc_optimizer == 'adam':
        disc_optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        
    else:  # rmsprop
        print('Optimizer not supported \n')
        return None


    if args.training_method == 'train_FDA':
        if val_only:
            train_FDA.val(args, model, dataloader_val, device)
        else:
            train_FDA.train(args, model, optimizer, dataloader_train, dataloader_target, dataloader_val, device)  ## train loop
            train_FDA.val(args, model, dataloader_val, device)                                              # final test
            
    if args.training_method == 'train_ADA':
        if val_only:
            train_ADA.val(args, model, dataloader_val, device)    
        else: 
            train_ADA.train(args, model, optimizer, disc_optimizer, dataloader_train, dataloader_target, dataloader_val, device)      ## train loop
            train_ADA.val(args, model, dataloader_val, device)                                                                          # final test

    else: #using standard training method
        if val_only:
            train_1.val(args, model, dataloader_val, device)
        else:
            train_1.train(args, model, optimizer, dataloader_train, dataloader_val, device)        ## train loop
            train_1.val(args, model, dataloader_val, device)                                        # final test



if __name__ == "__main__":
    main()
    
    # TODO: move evaluation function to separate file as we always use the same function
    # TODO: hide the dataloader and the preprocessing in a function
    # TODO: set the datasets based on the training method
    # TODO: revise the arguments probably val_only training_method and mode are not needed