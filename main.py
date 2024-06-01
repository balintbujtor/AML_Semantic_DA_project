import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import trainings.train_simple as train_simple
import trainings.train_ADA as train_ADA
import trainings.train_FDA as train_FDA
from trainings.val import val
from utils.utils import *
from utils.fda import test_multi_band_transfer, pseudo_label_gen
from model.model_stages import BiSeNet
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5


def main():
    
    # Fixing the random seeds
    random_seed = 42
    np.random.seed(random_seed)
    os.environ['SEED'] = str(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
       
    args = parse_args()

    # Handling checkpoint saves in a sub-folder
    save_keyword = args.save_keyword
    save_model_path = args.save_model_path
    save_subdir_path = make_saveDir(save_model_path,save_keyword)
    val_only = True if args.validation_only else False
    aug_method = args.aug_method
    
    action = args.action
    
    train_dataset = None
    target_dataset = None
    val_dataset = None
    
    num_classes = 19
    is_pseudo = False
    
    # selecting the datasets based on the action
    match action:
        
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
        
        case 'generate_pseudo_labels':
            train_dataset = 'gta5'
            target_dataset = 'cityscapes'
            val_dataset = 'cityscapes'
            is_pseudo = False
            
        case 'train_ssl_fda':
            train_dataset = 'gta5'
            target_dataset = 'cityscapes'
            val_dataset = 'cityscapes'
            is_pseudo = True

        case _:
            print('Training method not supported \n')


    if train_dataset == 'cityscapes':
        print("dataloader_train is on cityscapes")
        train_dataset = CityScapes(aug_method=aug_method, split='train', is_pseudo=False)
        dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)
        
    if target_dataset == 'cityscapes':
        print("dataloader_target is on cityscapes")
        # TODO: check if the is_pseudo argument is correctly set here for SSL FDA
        target_dataset = CityScapes(aug_method='', split='train', is_pseudo=is_pseudo)
        dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    if val_dataset == 'cityscapes':
        print("dataloader_val is on cityscapes")
        val_dataset = CityScapes(aug_method='', split='val', is_pseudo=False)
        dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
        

    if 'gta5' in (train_dataset,target_dataset,val_dataset):
        
        # TODO: (low prio) move this to the GTA5 class
        directory = "GTA5/images"
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
        if train_dataset == 'gta5':
            print("dataloader_train is on gta5")
            train_dataset = GTA5(aug_method=aug_method, training_method=action)
            dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True, sampler=train_sampler)
            
        if target_dataset == 'gta5':
            print("dataloader_target is on gta5")
            
            target_dataset = GTA5(aug_method='', training_method=action)
            dataloader_target = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=True, sampler=train_sampler)

        if val_dataset == 'gta5':
            print("dataloader_val is on gta5")
            val_dataset = GTA5(aug_method='', training_method=action)
            dataloader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False, sampler=valid_sampler)


    # model
    model = BiSeNet(backbone='CatmodelSmall', n_classes=19, pretrain_model=args.pretrain_path, use_conv_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model).cuda()

    # optimizer
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

    # calling function to perform requested action
    if action == 'train_simple_cityscapes' or action == 'train_simple_gta5':
        if val_only:
            model.load_state_dict(torch.load(args.load_model_path))
            val(model, dataloader_val, device, num_classes)
        else:
            train_simple.train(args, model, optimizer, dataloader_train, dataloader_val, num_classes, device, save_subdir_path, save_keyword)        ## train loop
            val(model, dataloader_val, device, num_classes)
            
    elif action == 'val_gta5_transfer':
        model.load_state_dict(torch.load(args.load_model_path))
        val(model, dataloader_val, device, num_classes)
    
    elif action == 'train_ada':
        if val_only:
            model.load_state_dict(torch.load(args.load_model_path))
            val(model, dataloader_val, device, num_classes)
        else: 
            train_ADA.train(args, model, optimizer, disc_optimizer, dataloader_train, dataloader_target, dataloader_val, num_classes, device, save_subdir_path, save_keyword)      ## train loop
            val(model, dataloader_val, device, num_classes)
              
    elif action == 'train_fda':
        if val_only:
            model.load_state_dict(torch.load(args.load_model_path))
            val(model, dataloader_val, device, num_classes)
        else:
            train_FDA.train(args, model, optimizer, dataloader_train, dataloader_target, dataloader_val, num_classes, device, beta=args.fda_beta)  ## train loop
            val(model, dataloader_val, device, num_classes)                                            # final test
    
    elif action == 'val_mbt':
        cp_model1 = "fill_me" # TODO
        cp_model2 = "fill_me"
        cp_model3 = "fill_me"
        precision, miou = test_multi_band_transfer(args, dataloader_val, num_classes, cp_model1, cp_model2, cp_model3, device)
        print(f"Precision: {precision}, mIoU: {miou}")

    elif action == 'generate_pseudo_labels':
        cp_model1 = './checkpoints/fda_beta05/best.pth' 
        cp_model2 = './checkpoints/fda_beta05/best.pth' 
        cp_model3 = './checkpoints/fda_beta05/best.pth' 
        precision, miou = pseudo_label_gen(args, dataloader_val, cp_model1, cp_model2, cp_model3, device)
        print(f"Precision: {precision}, mIoU: {miou}")
        
    elif action == 'train_ssl_fda':
        # TODO
        print('you forgot to do me')
        
    else:
        print('Training method not supported \n')
        return None
        
if __name__ == "__main__":
    
    # TODO: train FDA 3x with different betas
    # TODO: upload the trained models
    
    # TODO: debug the mbt
    # TODO: debug pseudo generation
    # TODO: debug the SSL FDA
    
    # TODO: perfrom MBT
    # TODO: generate pseudo labels
    # TODO: train SSL FDA
    
    main()

