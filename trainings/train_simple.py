#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import numpy as np
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils.utils import *
from tqdm import tqdm
from trainings.val import val


logger = logging.getLogger()


def train(args, model, optimizer, dataloader_train, dataloader_val, num_classes, device, save_subdir_path, save_keyword):
    """
    Simple training function for the initial STDC model.

    Args:
        args (_type_): Arguments that are specified in the command line when launching the main script.
        model (_type_): The model that is being trained.
        optimizer (_type_): the optimizier used for training. (Adam)
        dataloader_train (_type_): The dataloader for the training dataset.
        dataloader_val (_type_): The dataloader for the validation dataset.
        device (_type_): The device to train on, either cuda or cpu
        save_subdir_path (_type_): The path to save the model checkpoints.
        save_keyword (_type_): Keyword to save the model with.
    """
    max_miou = 0
    step = 0

    writer = SummaryWriter(comment=''.format(args.optimizer))
    scaler = amp.GradScaler()
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)

    
    for epoch in range(args.num_epochs):
        
        loss_record = []
        
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        
        # tqdm prints the training status and information on the terminal
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))

        for i, (data, label) in enumerate(dataloader_train):
            data = data.to(device)
            label = label.long().to(device)
            optimizer.zero_grad()

            # Automatic Mixed Precision (AMP) is used to speed up training
            # it lowers the precision of the weights and gradients to float16
            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

            # Backward and optimize, scalerr helps to avoid underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
            
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        # Save the model every checkpoint_step epochs
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            saveName = save_keyword + '-latest'
            save_checkpoint(model,save_subdir_path,saveName,includeTimestamp=False)

        # Validate the model every validation_step epochs
        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(model, dataloader_val, device, num_classes)
            
            # Save the model if the mIoU is the best so far
            if miou > max_miou:
                max_miou = miou
                saveName = save_keyword + '-best'
                save_checkpoint(model,save_subdir_path,saveName,includeTimestamp=False) 
                      
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
