import torch
from utils.utils import *
from datasets.cityscapes import CityScapes

def visualize(model, dataloader, device, num_classes, action, savepath):
    """
    Visualization function for the validation dataset.
    It saves the best accuracy and mIoU images, as well as a random image from the dataset.
    Extension of the val function
    IMPORTANT: it is run with batch size 1

    Args:
        model (_type_): The model that is being trained.
        dataloader (_type_): The dataloader for the validation dataset.
        device (_type_): The device to train on, either cuda or cpu
        num_classes (int): number of classes used
        action (str): The action to perform, to append to savepath
        savepath (str): The path to save the visualizations.

    """
    print('start of visualization!')
    with torch.no_grad():
        model.eval()

        max_miou = 0
        miou_img = 0
        max_miou_image = 0
        max_miou_label = 0
        max_precision = 0
        max_precision_image = 0
        max_precision_label = 0

        rand_idx = np.random.randint(0, len(dataloader))
        
        for i, (data, label) in enumerate(dataloader):
            label = label.type(torch.LongTensor)

            data = data.float().to(device)
            label = label.long().to(device)

            # get RGB predict image
            predict, _, _ = model(data)
            output = predict
            label_to_vis = label
            
            predict = predict.squeeze(0)
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label = label.squeeze()
            label = np.array(label.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            hist_img = fast_hist(label.flatten(), predict.flatten(), num_classes)
            miou_list_img = per_class_iu(hist_img)
            miou_img = np.mean(miou_list_img)
            
            if i == rand_idx:
                rd_color_pred, rd_color_lbl = CityScapes.visualize_prediction(output, label_to_vis.squeeze(1))
                rd_color_pred.save(os.path.join(savepath, action + 'random_color_pred.png'))
                rd_color_lbl.save(os.path.join(savepath, action + 'random_color_lbl.png'))
            
            # if 
            if precision > max_precision:
                max_precision = precision
                max_precision_image = output
                max_precision_label = label_to_vis.squeeze(1)

            if miou_img > max_miou:
                max_miou = miou_img
                max_miou_image = output
                max_miou_label = label_to_vis.squeeze(1)

        #Generate visuals of best accuracy/miou images
        color_precision_pred,color_precision_lbl = CityScapes.visualize_prediction(max_precision_image, max_precision_label)
        color_miou_pred,color_miou_lbl = CityScapes.visualize_prediction(max_miou_image, max_miou_label)

        # Save color precision prediction and label images
        color_precision_pred.save(os.path.join(savepath, action + 'color_precision_pred.png'))
        color_precision_lbl.save(os.path.join(savepath, action + 'color_precision_lbl.png'))
        
        # Save color miou prediction and label images
        color_miou_pred.save(os.path.join(savepath, action + 'color_miou_pred.png'))
        color_miou_lbl.save(os.path.join(savepath, action + 'color_miou_lbl.png'))
