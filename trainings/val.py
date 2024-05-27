import torch
from utils.utils import *


def val(model, dataloader, device, num_classes):
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