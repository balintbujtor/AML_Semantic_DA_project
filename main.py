import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import trainings.train_simple as train_simple
import trainings.train_ADA as train_ADA
import trainings.train_FDA as train_FDA
import trainings.val as val
from utils.utils import *
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5


def main():
    args = parse_args()

    # Handling checkpoint saves in a sub-folder
    save_keyword = args.save_keyword
    save_model_path = args.save_model_path
    save_subdir_path = make_saveDir(save_model_path,save_keyword)

    val_only = True if args.validation_only else False

    aug_method = args.aug_method


    # Fixing the random seeds
    random_seed = 42
    np.random.seed(random_seed)
    os.environ['SEED'] = str(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


    training_method = args.training_method
    
    match training_method:
        
        case 'train_simple_cityscapes':
            train_dataset = 'cityscapes'
            val_dataset = 'cityscapes'
            
        case 'train_simple_gta5':
            train_dataset = 'gta5'
            val_dataset = 'gta5'
            
        case 'val_gta5_transfer':
            val_dataset = 'cityscapes'
            
        case 'train_ada':
            train_dataset = 'gta5'
            target_dataset = 'cityscapes'
            val_dataset = 'cityscapes'
               
        case 'train_fda':
            train_dataset = 'gta5'
            target_dataset = 'cityscapes'
            val_dataset = 'cityscapes'          
        
        case 'val_mbt':
            val_dataset = 'cityscapes' 
            
        case 'train_ssl_fda':
            train_dataset = 'gta5'
            target_dataset = 'cityscapes'
            val_dataset = 'cityscapes'
     

    if train_dataset == 'cityscapes':
        print("dataloader_train is on cityscapes")
        train_dataset = CityScapes(aug_method=aug_method, split='train')
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
        
    if target_dataset == 'cityscapes':
        print("dataloader_target is on cityscapes")
        target_dataset = CityScapes(aug_method='', split='train')
        dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    if val_dataset == 'cityscapes':
        print("dataloader_val is on cityscapes")
        val_dataset = CityScapes(aug_method='', split='val')
        dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
        

    if 'gta5' in (train_dataset,target_dataset,val_dataset):
        
        directory = "GTA5\images"
        dataset_size = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
        indices = list(range(dataset_size))
        validation_split = 0.2
        split = int(np.floor(validation_split * dataset_size))
        
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # Setting the dataloaders
        if args.training_dataset == 'gta5':
            print("dataloader_train is on gta5")
            train_dataset = GTA5(aug_method=aug_method, training_method=training_method)
            dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True, sampler=train_sampler)
            
        if args.target_dataset == 'gta5':
            print("dataloader_target is on gta5")
            
            target_dataset = GTA5(aug_method='', training_method=training_method)
            dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True, sampler=train_sampler)

        if val_dataset == 'gta5':
            print("dataloader_val is on gta5")
            val_dataset = GTA5(aug_method='', training_method=training_method)
            dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False, sampler=valid_sampler)


    ## model
    model = BiSeNet(backbone='CatmodelSmall', n_classes=19, pretrain_model=args.pretrain_path, use_conv_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model).cuda()

    ## optimizer
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
        disc_optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
        disc_optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
     
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        disc_optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        
    else:
        print('Optimizer not supported \n')
        return None


    if args.training_method == 'train_FDA':
        if val_only:
            val(args, model, dataloader_val, device)
        else:
            train_FDA.train(args, model, optimizer, dataloader_train, dataloader_target, dataloader_val, device, beta=args.fda_beta)  ## train loop
            val(args, model, dataloader_val, device)                                              # final test
            
    if args.training_method == 'train_ADA':
        if val_only:
            val(args, model, dataloader_val, device)    
        else: 
            train_ADA.train(args, model, optimizer, disc_optimizer, dataloader_train, dataloader_target, dataloader_val, device, save_subdir_path, save_keyword)      ## train loop
            val(args, model, dataloader_val, device)                                                                          # final test

    else: #using standard training method
        if val_only:
            val(args, model, dataloader_val, device)
        else:
            train_simple.train(args, model, optimizer, dataloader_train, dataloader_val, device, save_subdir_path, save_keyword)        ## train loop
            val(args, model, dataloader_val, device)                                        # final test



if __name__ == "__main__":
    
    # TODO: train FDA 3x with different betas
    
    # TODO: debug the mbt
    # TODO: debug pseudo generation
    
    # TODO: 3 join cityscapes and cityscapes ssl into 1
    # TODO: debug
    
    # TODO: debug the refactored code
    
    # TODO: comment the code

    main()

