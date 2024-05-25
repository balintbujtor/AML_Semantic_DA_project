from PIL import Image
import torch
import os
import numpy as np
from model.model_stages import BiSeNet
import torch.nn as nn
import torch.nn.functional as F

## FDA
# we are using the np version because they are working without major modifications to the original code

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

## Loss function FDA
## Weighting function for entropy minimization
class EntropyLoss(nn.Module):
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