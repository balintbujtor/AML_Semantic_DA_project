from calendar import c
from email.mime import image
import torch
from datasets.cityscapes import CityScapes
from utils.utils import *


def visualize(model, dataset, dataloader, device, num_classes, savepath):
    """
    Validation function for the models across all training scripts.

    Args:
        model (_type_): The model that is being trained.
        dataloader (_type_): The dataloader for the validation dataset.
        device (_type_): The device to train on, either cuda or cpu
        num_classes (int): number of classes used

    Returns:
        the precision and mIoU for the validation dataset.
    """
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((num_classes, num_classes))

        max_miou = 0
        max_miou_image = 0
        max_miou_label = 0
        max_precision = 0
        max_precision_image = 0
        max_precision_label = 0

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
            hist += fast_hist(label.flatten(), predict.flatten(), num_classes)

            # if 
            if precision > max_precision:
                max_precision = precision
                max_precision_image = predict
                max_precision_label = label

            if miou > max_miou:
                max_miou = miou
                max_miou_image = predict
                max_miou_label = label

            precision_record.append(precision)

        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)
        miou = np.mean(miou_list)

        #Generate visuals of best accuracy/miou images
        color_precision_pred,color_precision_lbl = dataset.visualize_prediction(dataset,max_precision_image,max_precision_label)
        color_miou_pred,color_miou_lbl = dataset.visualize_prediction(dataset,max_miou_image,max_miou_label)

        
        # Save color precision prediction and label images
        color_precision_pred.save(os.join(savepath,'color_precision_pred.png'))
        color_precision_lbl.save(os.join(savepath,'color_precision_lbl.png'))
        
        # Save color miou prediction and label images
        color_miou_pred.save(os.join(savepath,'color_miou_pred.png'))
        color_miou_lbl.save(os.join(savepath,'color_miou_lbl.png'))


        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        print(f'mIoU per class: {miou_list}')

        return precision, miou