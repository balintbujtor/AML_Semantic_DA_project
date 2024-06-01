from email.mime import image
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model.model_stages import BiSeNet
import utils.utils as ut


"""
Functions related to the FDA method.
Functions are taken from the following repository:
    https://github.com/YanchaoYang/FDA

We use the np versions because they work without major modifications to the original code
"""

def low_freq_mutate_np( amp_src, amp_trg, L=0.1 ):
    a_src = np.fft.fftshift( amp_src, axes=(-2, -1) )
    a_trg = np.fft.fftshift( amp_trg, axes=(-2, -1) )

	# extrd dim for the batch
    _, _, h, w = a_src.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

	# extra dim for the batch
    a_src[:,:,h1:h2,w1:w2] = a_trg[:,:,h1:h2,w1:w2]
    a_src = np.fft.ifftshift( a_src, axes=(-2, -1) )
    return a_src

def FDA_source_to_target_np( src_img, trg_img, L=0.1 ):
    # exchange magnitude
    # input: src_img, trg_img

    src_img_np = src_img #.cpu().numpy()
    trg_img_np = trg_img #.cpu().numpy()

    # get fft of both source and target
    fft_src_np = np.fft.fft2( src_img_np, axes=(-2, -1) )
    fft_trg_np = np.fft.fft2( trg_img_np, axes=(-2, -1) )

    # extract amplitude and phase of both ffts
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)
    amp_trg, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    # mutate the amplitude part of source with target
    amp_src_ = low_freq_mutate_np( amp_src, amp_trg, L=L )

    # mutated fft of source
    fft_src_ = amp_src_ * np.exp( 1j * pha_src )

    # get the mutated image
    src_in_trg = np.fft.ifft2( fft_src_, axes=(-2, -1) )
    src_in_trg = np.real(src_in_trg)

    return src_in_trg


class EntropyLoss(nn.Module):
    """
    Loss function FDA
    Weighting function for entropy minimization

    """
    
    def __init__(self):
        super(EntropyLoss, self).__init__()
	
    def forward(self, x, ita):
        P = F.softmax(x, dim=1)        # [B, 19, H, W]
        logP = F.log_softmax(x, dim=1) # [B, 19, H, W]
        PlogP = P * logP               # [B, 19, H, W]
        ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
        ent = ent / 2.9444 # chanage when classes is not 19
        # compute robust entropy
        ent = ent ** 2.0 + 1e-8
        ent = ent ** ita
        ent_loss_value = ent.mean()
        
        return ent_loss_value


def test_multi_band_transfer(args, 
                            dataloader_target_val, 
                            model1_path, 
                            model2_path, 
                            model3_path, 
                            device,
                            num_classes: int = 19
                            ):
    """
    Test the performance of 3 different models on the validation dataset.
    The predicted output is the average output of the 3 models.

    Args:
        args (_type_): Arguments that are specified in the command line when launching the main script.
        dataloader_target_val (_type_): The dataloader for the validation dataset.
        model1_path (_type_): path for the first model
        model2_path (_type_): path for the second model
        model3_path (_type_): path for the third model
        device (_type_): The device to train on, either cuda or cpu

    Returns:
        the precision and mIoU of the 3 models for the validation dataset.
    """
    
    hist = np.zeros((num_classes, num_classes))
    precision_record = []
    
    # load the models and set them to evaluation mode
    backbone='CatmodelSmall'
    model1 = BiSeNet(backbone, n_classes = num_classes,use_conv_last=False)
    model1.load_state_dict(torch.load(model1_path))
    model1.eval()
    model1.to(device)

    model2 = BiSeNet(backbone, n_classes = num_classes,use_conv_last=False)
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()
    model2.to(device)
    
    model3 = BiSeNet(backbone, n_classes = num_classes,use_conv_last=False)
    model3.load_state_dict(torch.load(model3_path))
    model3.eval()
    model3.to(device)

    with torch.no_grad():
        for i, (data, label, _) in enumerate(dataloader_target_val):
            
            data = data.to(device)
            label = label.long.to(device)
            
            # take the predictions of the 3 models and average them
            pred_1, _, _ = model1(data)
            pred_2, _, _ = model2(data)
            pred_3, _, _ = model3(data)

            pred = (pred_1 + pred_2 + pred_3) / 3

            pred = pred.squeeze(0)
            pred = ut.reverse_one_hot(pred)
            pred = np.array(pred.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = ut.compute_global_accuracy(pred, label)
            hist += ut.fast_hist(label.flatten(), pred.flatten(), num_classes)

            # there is no need to transform the one-hot array to visual RGB array
            # pred = colour_code_segmentation(np.array(pred), label_info)
            # label = colour_code_segmentation(np.array(label), label_info)
            
            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = ut.per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou

def pseudo_label_gen(args, 
                     dataloader_target_val,
                     model1_path, 
                     model2_path, 
                     model3_path,
                     device,
                     save_path: str = 'Cityscapes/Cityspaces/pseudo_labels',
                     num_classes: int = 19
                     ):
    """_summary_

    Args:
        args (_type_): Arguments that are specified in the command line when launching the main script.
        dataloader_target_val (_type_): The dataloader for the validation dataset.
        model1_path (_type_): path for the first model
        model2_path (_type_): path for the second model
        model3_path (_type_): path for the third model
        device (_type_): The device to train on, either cuda or cpu
        save_path (str, optional): The paths where to save the generated pseudo labels. 
                                   Defaults to 'Cityscapes/Cityspaces/pseudo_labels'.
    """

    # 'train' or 'val'
    split = dataloader_target_val.dataset.split

    # Computation of Precision
    hist = np.zeros((num_classes,num_classes))
    precision_record = []
    
    # create the folder if it does not exist
    save_path_w_mode = os.path.join(save_path, split)
    if not os.path.exists(save_path_w_mode):
        os.makedirs(save_path_w_mode)
    
    # load the models and set them to evaluation mode
    backbone='CatmodelSmall'
    model1 = BiSeNet(backbone, n_classes = num_classes,use_conv_last=False)
    model1.load_state_dict(torch.load(model1_path))
    model1.eval()
    model1.to(device)

    model2 = BiSeNet(backbone, n_classes = num_classes,use_conv_last=False)
    model2.load_state_dict(torch.load(model2_path))
    model2.eval()
    model2.to(device)
    
    model3 = BiSeNet(backbone, n_classes = num_classes,use_conv_last=False)
    model3.load_state_dict(torch.load(model3_path))
    model3.eval()
    model3.to(device)

    predicted_label = np.zeros((len(dataloader_target_val), 512,1024), dtype=np.uint8)
    predicted_prob = np.zeros((len(dataloader_target_val), 512,1024), dtype=np.float32)    
    image_names = []
    
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader_target_val):
            
            data = data.to(device)
            label = label.long.to(device)
            
            # generate the predictions and average them to get the pseudo label
            pred_1, _, _ = model1(data)
            pred_2, _, _ = model2(data)
            pred_3, _, _ = model3(data)

            pred = (pred_1 + pred_2 + pred_3) / 3
            pred = torch.nn.functional.softmax(pred, dim=1)
            
            label, prob = np.argmax(pred, axis=2), np.max(pred, axis=2)
            predicted_label[i] = label.copy()
            predicted_prob[i] = prob.copy()

            # compute per pixel accuracy
            precision = ut.compute_global_accuracy(pred, label)
            hist += ut.fast_hist(label.flatten(), pred.flatten(), num_classes)

            precision_record.append(precision)
            
            # go through the images in the batch to save the image names
            for j in range(data.size(0)):
                
                # TODO: check if it is correct
                image_path = dataloader_target_val.dataset.image_paths[i*args.batch_size + j]
                                
                # the first split keeps only the name of the image, the second split removes the extension
                image_name = image_path.split('/')[-1].split('.')[0]
                
                # remove the last part of the name, i.e. '_leftImg8bit'
                image_name = image_name.rsplit('_', 1)[0]
                
                # e.g. 'hanover_000000_000019_psedudo_labelTrainIds.png'
                image_name = image_name + '_pseudo_labelTrainIds.png'
                image_names.append(image_name)

        # precision
        precision = np.mean(precision_record)
        miou_list = ut.per_class_iu(hist)
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')        

    # Each class has a threshold depending on the frequency of it 
    # So for each label we check if it's less than the threshold of the corresofing class this label is ignored
    # Otherwise each pixel will be given the label with highest proba even when the probe is really small -> accuracy drops

    thres = []
    for i in range(19):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.66))])
    print( thres )
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    print( thres )
    
    # go through the dataloader and save the pseudo labels
    for index in range(len(dataloader_target_val)):
        name = image_names[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        
        # set the label to 255 if the probability is less than the threshold
        for i in range(19):
            label[   (prob<thres[i]) * (label==i)   ] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        
        # get the city name --> for the folder
        city = name.split('_')[0]
        
        # e.g. 'Cityscapes/Cityspaces/pseudo_labels/val/hanover/'
        save_path = os.path.join(save_path_w_mode, city)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        output.save('%s/%s' % (save_path, name)) 

    return precision, miou