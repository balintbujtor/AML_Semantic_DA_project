#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import numpy as np
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
from torchvision.transforms import v2

logger = logging.getLogger()

# computed mean and std of the Cityscapes dataset, on our dataset with our function
# ???: decide if we want to use this or the imagenet mean and std
CS_MEAN = torch.tensor([0.3075, 0.3437, 0.3014])
CS_STD = torch.tensor([0.1880, 0.1908, 0.1881])

normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def val(args, model, dataloader, device):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)

            data = data.to(device)
            label = label.long().to(device)

            # get RGB predict image
            predict, _, _ = model(data)
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # predict = colour_code_segmentation(np.array(predict), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou




    # We will use both GTA5 and Cityscapes for the training
    #   dataloader_source is the dataloader of GTA5
    #   data_loader_target is the dataloader of Cityscapes
    # We validate only on Cityscapes


def train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, device, beta=0.09, ita=2, entW=0.005):
    
    max_miou = 0
    step = 0
    
    writer = SummaryWriter(comment=''.format(args.optimizer))
    # to handle small gradients and avoid vanishing gradients
    scaler = amp.GradScaler()
    
    ## learning rate of the segmentation model
    learning_rate = args.learning_rate
   
    ## -- Loss functions --
    ## loss function for the segmentation model
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_entr = EntropyLoss()
    
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, learning_rate, iter=epoch, max_iter=args.num_epochs)

        ## Training loop
        model.train()

        tq = tqdm(total=min(len(dataloader_source),len(dataloader_target)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr ))
        
        ## Losses for the segmentation model
        loss_record = []
        
        for i, ((data_source, label), (data_target, _)) in enumerate(zip(dataloader_source, dataloader_target)):


            # 1. Source to Target, Target to Target : Adapt source image to target image
            source_in_target = FDA_source_to_target_np(data_source, data_target, L=beta)
            source_in_target = torch.from_numpy(source_in_target).float()
            target_in_target = data_target

            # 2. Subtract the mean / normalize it               
            source_image = normalize(source_in_target.clone()).cuda()
            target_image = normalize(target_in_target.clone()).cuda()
            label = label.long().cuda()
         

            ## clearing the gradients of all optimized variables. This is necessary 
            ## before computing the gradients for the current batch, 
            ## as you don't want gradients from previous iterations affecting the current iteration.
            optimizer.zero_grad()
            
            # Train segmentation
            # Loss for segmentation : Train on Source
            with amp.autocast():
                source_output, source_out16, source_out32 = model(source_image)
                target_output, _, _ = model(target_image)
                
                loss1 = loss_func(source_output, label.squeeze(1))
                loss2 = loss_func(source_out16, label.squeeze(1))
                loss3 = loss_func(source_out32, label.squeeze(1))
                
                loss_source = loss1 + loss2 + loss3
                loss_target = loss_entr(target_output, ita)
                
            triger_ent = 0.0
            # at a certain value of the epoch add the entropy minimization function
            if epoch > args.switch2entropy:
                triger_ent = 1.0

            # Total loss
            loss_total = loss_source + triger_ent * loss_target *entW

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss_total)
            step += 1
            writer.add_scalar('loss_step', loss_total, step)
            loss_record.append(loss_total.item())
            
        tq.close()
        loss_train_mean = np.mean(loss_record)

        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))


        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, device)
            if miou > max_miou:
                max_miou = miou          
                import os,datetime
                os.makedirs(args.save_model_path, exist_ok=True)
                timestamp = datetime.datetime.now().strftime('%Y-%m-%dZ%H:%M:%S') + '.pth'
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best'+timestamp+'.pth'))

            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
