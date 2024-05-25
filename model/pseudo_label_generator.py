from PIL import Image
import torch
import os
import numpy as np
from model.model_stages import BiSeNet

def pseudo_label_gen(args, 
                     dataloader_target_val,
                     checkpoint1_path, 
                     checkpoint2_path, 
                     checkpoint3_path,
                     save_path,
                     device):
    ## We need the weights from training FDA on different Betas
    ## let's say we're usind 3 betas as the repo

    # TODO check
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    backbone = args.backbone
    model1 = BiSeNet(backbone, n_classes = args.num_classes,use_conv_last=args.use_conv_last )
    model1.load_state_dict(torch.load(checkpoint1_path))
    model1.eval()
    model1.to(device)

    model2 = BiSeNet(backbone, n_classes = args.num_classes,use_conv_last=args.use_conv_last )
    model2.load_state_dict(torch.load(checkpoint2_path))
    model2.eval()
    model2.to(device)
    
    model3 = BiSeNet(backbone, n_classes = args.num_classes,use_conv_last=args.use_conv_last )
    model3.load_state_dict(torch.load(checkpoint3_path))
    model3.eval()
    model3.to(device)

    predicted_label = np.zeros((len(dataloader_target_val), 512,1024), dtype=np.uint8)
    predicted_prob = np.zeros((len(dataloader_target_val), 512,1024), dtype=np.float32)    
    image_names = []
    
    with torch.no_grad():
        for i, (data, label, name) in enumerate(dataloader_target_val):
            
            data = data.to(device)
            label = label.long.to(device)
            
            pred_1, _, _ = model1(data)
            pred_2 = model2(data)
            pred_3 = model3(data)

            pred = (pred_1 + pred_2 + pred_3) / 3
            pred = torch.nn.functional.softmax(pred, dim=1)
            
            label, prob = np.argmax(pred, axis=2), np.max(pred, axis=2)
            predicted_label[i] = label.copy()
            predicted_prob[i] = prob.copy()
            image_names.append(name[0])

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

    for index in range(len(dataloader_target_val)):
        name = image_names[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for i in range(19):
            label[   (prob<thres[i]) * (label==i)   ] = 255  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('/')[-1]
        output.save('%s/%s' % (args.save, name)) 


    return