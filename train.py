#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import logging
import numpy as np
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils import *
from tqdm import tqdm
from torchvision import transforms
from model.model_stages import BiSeNet
from cityscapes import CityScapes
from gta5 import GTA5

logger = logging.getLogger()


def val(args, model, dataloader, device):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)
            #maybe this is not good and cuda() is needed
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


def train(args, model, optimizer, dataloader_train, dataloader_val, device):
    writer = SummaryWriter(comment=''.format(args.optimizer))

    scaler = amp.GradScaler()

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label) in enumerate(dataloader_train):
            data = data.to(device)
            label = label.long().to(device)
            optimizer.zero_grad()

            with amp.autocast():
                output, out16, out32 = model(data)
                loss1 = loss_func(output, label.squeeze(1))
                loss2 = loss_func(out16, label.squeeze(1))
                loss3 = loss_func(out32, label.squeeze(1))
                loss = loss1 + loss2 + loss3

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
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            import os
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'latest.pth'))

        if epoch % args.validation_step == 0 and epoch != 0:
            precision, miou = val(args, model, dataloader_val, device)
            if miou > max_miou:
                max_miou = miou
                import os
                os.makedirs(args.save_model_path, exist_ok=True)
                torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'best.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)


def main():
    args = parse_args()

    ## dataset
    n_classes = args.num_classes

    root_dir = args.root_dir
    
    mode = args.mode
    chosen_dataset = args.dataset
    
    assert chosen_dataset in ['cityscapes', 'gta5'], "Dataset not supported"
    
    if chosen_dataset == 'cityscapes':
        train_dataset = CityScapes(root_dir=root_dir, mode=mode)
        dataloader_train = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)

        val_dataset = CityScapes(root_dir=root_dir, mode='val')
        dataloader_val = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)
        
    elif chosen_dataset == 'gta5':
        train_dataset = GTA5(root_dir=root_dir, mode=mode)
        dataloader_train = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
        
        val_dataset = GTA5(root_dir=root_dir, mode='val')
        dataloader_val = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    drop_last=False)

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
    
# NOTE: maybe the number of classes needs to be modified?