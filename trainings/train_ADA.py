#!/usr/bin/python
# -*- encoding: utf-8 -*-
from tkinter import Variable
import torch
import logging
import numpy as np
import torch.cuda.amp as amp
from tensorboardX import SummaryWriter
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
from model.discriminator import Discriminator

logger = logging.getLogger()


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


def train(args, model, optimizer, disc_optimizer, dataloader_source, dataloader_target, dataloader_val, device, save_keyword):
    
    # constants
    Lambda_adv = 0.0002
    
    
    writer = SummaryWriter(comment=''.format(args.optimizer))
    scaler = amp.GradScaler()
    
    ## learning rate of the segmentation model
    learning_rate = args.learning_rate

    ## learning rate of the dsicriminator
    disc_learning_rate = args.disc_learning_rate

    ## -- Loss functions --
    ## loss function for the segmentation model
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    ## loss function for the discriminator
    adv_loss_func  = torch.nn.BCEWithLogitsLoss()
    disc_loss_func = torch.nn.BCEWithLogitsLoss()
    # Labels for disc training
    label_source = 0
    label_target = 1

    max_miou = 0
    step = 0

    ## Initialize discriminator
    disc_model = torch.nn.DataParallel(Discriminator(num_classes=args.num_classes)).to(device) 
    disc_optimizer.zero_grad() # by default we use adam for both the segmentation model and the discriminator
    
    for epoch in range(args.epoch_start_i, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, learning_rate, iter=epoch, max_iter=args.num_epochs)
        disc_lr = poly_lr_scheduler(disc_optimizer, disc_learning_rate, iter=epoch, max_iter=args.num_epochs)

        ## Training loop
        model.train()
        disc_model.train()


        tq = tqdm(total=min(len(dataloader_source),len(dataloader_target)) * args.batch_size)
        tq.set_description('epoch %d, lr %f, lr_disc %f' % (epoch, lr, disc_lr))
        
        
        ## Losses for the segmentation model
        loss_record = []
        ## Losses for the discriminator model
        # disc_loss_record = []
        

        for ((data_source, label), (data_target, _)) in enumerate(zip(dataloader_source, dataloader_target)):
            data_source = data_source.to(device)
            label = label.long().to(device)
            data_target = data_target.to(device)
         

            ## clearing the gradients of all optimized variables. This is necessary 
            ## before computing the gradients for the current batch, 
            ## as you don't want gradients from previous iterations affecting the current iteration.
            optimizer.zero_grad()
            disc_optimizer.zero_grad() #recheck
            
            # Train segmentation
            
            # Training on the source
            # These parameters include the weights and biases of the layers.
            for param in disc_model.parameters():
                param.requires_grad = False


            with amp.autocast():
                source_output, source_out16, source_out32 = model(data_source)
                target_output, _, _ = model(data_target)
                
                loss1 = loss_func(source_output, label.squeeze(1))
                loss2 = loss_func(source_out16, label.squeeze(1))
                loss3 = loss_func(source_out32, label.squeeze(1))
                loss_segmentation = loss1 + loss2 + loss3
                loss = loss_segmentation

            scaler.scale(loss).backward()

            # reset the optimizer of segmentation model to zero
                # optimizer.zero_grad()

            ## Train with target data -- fooling 
            with amp.autocast():
                D_t_output = disc_model(F.softmax(target_output))
                
                loss_adv_target = adv_loss_func(D_t_output,
                                            torch.FloatTensor(D_t_output.data.size()).fill_(label_source).to(device))
                
            adv_loss = Lambda_adv*loss_adv_target
                
                                               
            
            scaler.scale(adv_loss).backward()
           

            ## Training the discriminator
            for param in disc_model.parameters():
                param.requires_grad = True

            source_output = source_output.detach()
            target_output = target_output.detach()

            ## Training on the source domain
            with amp.autocast():
                D_out1_s = disc_model(F.softmax(source_output))
                loss_D1_s = disc_loss_func(D_out1_s,
                                          torch.FloatTensor(D_out1_s.data.size()).fill_(label_source).to(device))
            
            scaler.scale(loss_D1_s).backward()
            
            ## Training on the traget domain
            with amp.autocast():
                D_out1_t = disc_model(F.softmax(target_output))
                loss_D1_t = disc_loss_func(D_out1_t,
                                          torch.FloatTensor(D_out1_t.data.size()).fill_(label_source).to(device))
            scaler.scale(loss_D1_t).backward()
            
            
            scaler.step(disc_optimizer)
            scaler.step(optimizer)
            scaler.update()
       

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss, loss_D1_s='%.6f' %(loss_D1_s/Lambda_adv), loss_D1_t='%.6f' %loss_D1_t)
            step += 1
            
            loss_record.append(loss.item())
            # disc_loss_record.append(loss_D1_t.item())
            
            # writer.add_scalar('loss_step', loss, step)
           
            
        tq.close()
        loss_train_mean = np.mean(loss_record)

        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            saveName = save_keyword + '-latest'
            saveName_disc = save_keyword + '-latest_disc'  
            save_checkpoint(model,args.save_model_path,saveName)
            save_checkpoint(disc_model,args.save_model_path,saveName_disc)


        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, device)
            if miou > max_miou:
                max_miou = miou
                saveName = save_keyword + '-best'
                saveName_disc = save_keyword + '-best_disc'          
                save_checkpoint(model,args.save_model_path,saveName)
                save_checkpoint(disc_model,args.save_model_path,saveName_disc)
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)
