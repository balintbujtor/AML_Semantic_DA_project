#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import os
import numpy as np
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.fda import *
from tqdm import tqdm
from torchvision.transforms import v2
from trainings.val import val


logger = logging.getLogger()
normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


# We will use both GTA5 and Cityscapes for the training
#   dataloader_source is the dataloader of GTA5
#   data_loader_target is the dataloader of Cityscapes => here the labels are the pseudo labels generated before
# We validate only on Cityscapes
def train(args, model, optimizer, dataloader_train_source, dataloader_train_target_SSL, dataloader_val, device, beta=0.09, ita=2, entW=0.005):
    
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

        tq = tqdm(total=min(len(dataloader_train_source),len(dataloader_train_target_SSL)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr ))
        
        ## Losses for the segmentation model
        loss_record = []
        
        for i, ((data_source, label_source), (data_target, label_target_pseudo)) in enumerate(zip(dataloader_train_source, dataloader_train_target_SSL)):


            # 1. Source to Target, Target to Target : Adapt source image to target image
            source_in_target = FDA_source_to_target_np(data_source, data_target, L=beta)
            source_in_target = torch.from_numpy(source_in_target).float()
            target_in_target = data_target

            # 2. Subtract the mean / normalize it               
            image_src = normalize(source_in_target.clone()).to(device)
            label_src = label_source.long().to(device)
            
            image_trg = normalize(target_in_target.clone()).to(device)
            label_trg_psu = label_target_pseudo.long().to(device)

            ## clearing the gradients of all optimized variables. This is necessary 
            ## before computing the gradients for the current batch, 
            ## as you don't want gradients from previous iterations affecting the current iteration.
            optimizer.zero_grad()

            with amp.autocast():
                
                # Predictions and loss on source images
                source_output, source_out16, source_out32 = model(image_src)
                loss1 = loss_func(source_output, label_src.squeeze(1))
                loss2 = loss_func(source_out16, label_src.squeeze(1))
                loss3 = loss_func(source_out32, label_src.squeeze(1))
                loss_seg_source = loss1 + loss2 + loss3
                
                # Predictions and loss on target images
                trg_output, trg_out16, trg_out32 = model(image_trg)
                
                # Compute the entropy loss on the target
                loss_ent = loss_entr(trg_output, ita)

                # Compute the CE loss with the pseudo labels
                loss1 = loss_func(trg_output, label_trg_psu.squeeze(1))
                loss2 = loss_func(trg_out16, label_trg_psu.squeeze(1))
                loss3 = loss_func(trg_out32, label_trg_psu.squeeze(1))
                loss_seg_pseudo = loss1 + loss2 + loss3            

                # Total loss
                loss_total = loss_seg_source + loss_seg_pseudo + entW*loss_ent

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_train += loss_seg_source.detach().cpu().numpy()
            loss_val += loss_seg_pseudo.detach().cpu().numpy()
            
            # TODO
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
