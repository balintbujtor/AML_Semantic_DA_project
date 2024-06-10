#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import numpy as np
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.fda import *
from tqdm import tqdm
from trainings.val import val
from utils.transforms import v2Normalize

logger = logging.getLogger()


def train(args, model, optimizer, dataloader_source, dataloader_target, dataloader_val, num_classes, device, beta=0.09, ita=2, entW=0.005):
    """
    Training function for Fourier Domain Adaptation (FDA).

    Args:
        args (_type_): Arguments that are specified in the command line when launching the main script.
        model (_type_): The model that is being trained.
        optimizer (_type_): the optimizier used for training. (Adam)
        dataloader_train (_type_): The dataloader for the training dataset. (GTA5)
        dataloader_val (_type_): The dataloader for the validation dataset. (Cityscapes)
        device (_type_): The device to train on, either cuda or cpu
        beta (float, optional): Hyperparameter that controls the low-freq windows size to be swapped.
                                Defaults to 0.09.
        ita (int, optional): coefficient for the Charbonnier penalty. Defaults to 2.
        entW (float, optional): weight of the entropy minimization loss. Defaults to 0.005.
    """
    max_miou = 0
    step = 0
    
    writer = SummaryWriter(comment=''.format(args.optimizer))
    # to handle small gradients and avoid vanishing gradients
    scaler = amp.GradScaler()
    
    # learning rate of the  model
    learning_rate = args.learning_rate
   
    # loss functions
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    loss_entr = EntropyLoss()
    
    for epoch in range(args.num_epochs):
        
        loss_record = []
        
        lr = poly_lr_scheduler(optimizer, learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()

        # tqdm prints the training status and information on the terminal
        tq = tqdm(total=min(len(dataloader_source),len(dataloader_target)) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr ))
        
        for i, ((data_source, label_source), (data_target, _)) in enumerate(zip(dataloader_source, dataloader_target)):


            data_source = data_source.to(device)
            label_source = label_source.long().to(device)

            data_target = data_target.to(device)

            orig_data_source =  data_source.clone()
            orig_data_target = data_target.clone()

            source_in_target = FDA_source_to_target(data_source, data_target, L=beta)
            
            # Normalize the source and target images
            source_in_target = torch.clamp(source_in_target, 0, 255) # / 255.0
            source_in_target = v2Normalize(source_in_target)
            
            data_target = torch.clamp(data_target, 0, 255) # data_target / 255.0
            data_target = v2Normalize(data_target)
            
            # Clearing the gradients of all optimized variables.  
            # This is necessary before computing the gradients for the current batch, 
            # as you don't want gradients from previous iterations affecting the current iteration.
            optimizer.zero_grad()
            
            with amp.autocast():
                
                # Predict and compute the segmentation loss on the source domain
                source_output, source_out16, source_out32 = model(source_in_target)
                loss1 = loss_func(source_output, label_source.squeeze(1))
                loss2 = loss_func(source_out16, label_source.squeeze(1))
                loss3 = loss_func(source_out32, label_source.squeeze(1))
                loss_seg_source = loss1 + loss2 + loss3
                
                # Predict and compute the entropy minimization loss on the target domain
                target_output, target_out16, target_out32 = model(data_target)
                loss_target = loss_entr(target_output, ita)

            # Total loss
            loss_total = loss_seg_source + loss_target *entW

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
        
        # Save the model every checkpoint_step epochs
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        # Validate the model every validation_step epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val( model, dataloader_val, device, num_classes)
            
            if miou > max_miou:
                max_miou = miou          
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(),os.path.join(args.save_model_path, 'best.pth'))

            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
