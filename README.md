## About
This project tackles Domain Adaptation applied to the Real-time Semantic Segmentation
networks, featuring an Adversarial Domain Adaptation algorithm.

## TODO
- [x] Matching label categories of GTA5 with Cityscapes
- [ ] Training NN on Cityscapes and filling out table1

## Useful links
-  [Colabs env](https://colab.research.google.com/drive/1TDjhWjOQwZ8ToXjDGRF43G7Qk590C7jP)
- [Docs](https://drive.google.com/drive/folders/1_a1j7FWd2zgzU6ZLaQybO9f02mJM_Uyo)

## Results

Common parameters:
- Base model: STCD
- Epochs: 50
- Discriminator: Adam

| Experiment                                       | Accuracy (%) | mIoU (%) | Time (avg per-epoch) | saveFile |
| ------------------------------------------------ | ------------ | -------- | -------------------- | -------- |
| training & validation on Cityscapes              |     78.7     |  44.4    |                      |          |
| training & validation on GTA5                    |     78.2     |  49.5    |                      |          |
| Domain shift evaluation GTA5>Cityscapes          |     52.1     |  11.5    |                      |          |
| ADA GTA5>Cityscapes without data augmentation    |     59.4     |  19.3    |                      |          |
| ADA GTA5>Cityscapes with data augmentation       |     67.2     |  20.1    |                      |          |
| SSL FDA GTA5>Cityscapes with data augmentation   |              |          |                      |          |
| SSL FDA GTA5>Cityscapes without data augmentation|              |          |                      |          |

### Detailed Results
- **training & validation on Cityscapes:**

saveFile:
date:
precision per pixel for test: 0.787 
mIoU for validation: 0.444 
mIoU per class: [0.95377805 0.67910142 0.82291703 0.24003275 0.21085642 0.24693448 
 0.13421382 0.358522   0.84801198 0.44508914 0.88199773 0.48100367 
 0.20703901 0.82966513 0.15454098 0.22780025 0.13120532 0.1293023 
 0.46101077]
 
- **training & validation on GTA5:**

saveFile:
date:
precision per pixel for test: 0.782 
mIoU for validation: 0.495 
mIoU per class: [0.96391841 0.70049577 0.7932551  0.55256778 0.24198456 0.35509488 
 0.32561131 0.240652   0.74605463 0.6721074  0.94207433 0.19218172 
 0.16312223 0.64258123 0.53373726 0.44477375 0.61858683 0.24296685 
 0.02396857]

- **Domain shift evaluation GTA5>Cityscapes:**

date:
precision per pixel for test: 0.521
mIoU for validation: 0.115
mIoU per class: [7.61754694e-01 1.71559990e-02 2.95094636e-01 1.65790444e-05
 0.00000000e+00 5.45316307e-03 1.28892443e-03 0.00000000e+00
 2.86752054e-01 4.46510915e-02 4.82411080e-01 0.00000000e+00
 0.00000000e+00 2.91239577e-01 1.42408418e-03 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00]

- **ADA without data aug:**

saveFile:
date:
precision per pixel for test: 0.594
mIoU for validation: 0.193
mIoU per class: [6.81801228e-01 1.07932154e-01 6.66776069e-01 5.99369935e-02
 6.49155477e-02 1.30033783e-01 5.31078158e-02 6.19663108e-03
 6.49269879e-01 9.55536141e-02 6.63765725e-01 7.46360803e-02
 8.57146883e-03 3.10703607e-01 4.97428665e-02 1.42098133e-02
 4.24880613e-03 1.70739188e-02 2.84853135e-04]

- **ADA with data aug:**

  /!\ to redo, values for only 30 epochs currently
saveFile:
date: 26/05/2024
precision per pixel for test: 0.672
mIoU for validation: 0.201
mIoU per class: [8.55649679e-01 8.32756948e-02 6.50656282e-01 7.27817810e-02
 3.19655737e-03 8.41106085e-02 2.09969896e-02 2.44836400e-04
 6.60987629e-01 1.58780603e-01 6.76856284e-01 1.07935027e-02
 0.00000000e+00 5.04739004e-01 3.91791383e-02 2.96842603e-04
 1.00073053e-05 1.37119106e-03 0.00000000e+00]
  
  
  
