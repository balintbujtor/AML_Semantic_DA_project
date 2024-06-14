# Real-time Domain Adaptation in Semantic Segmentation

## About
This project tackles Domain Adaptation applied to the Real-time Semantic Segmentation
networks, featuring an Adversarial Domain Adaptation algorithm.

### Authors 
 - Bálint Bujtor
 - Boyan Cieutat
 - Inaam Elhelwe

## Project structure
  - `datasets/`: contains the classes that handle the datasets
    - [Cityscapes](datasets/cityscapes.py)
    - [GTA5](datasets/gta5.py)
  - `model/`: contains the implementation of the models and their components that are used in the project
    - [STDC backbone](model/stdcnet.py) - the backbone of the model
    - [BiSeNet](model/model_stages.py)
    - [Discriminator](model/discriminator.py) - discriminator used in the adversarial domain adaptation
  - `trainings/`: contains the training scripts
    - [Simple training](trainings/train_simple.py) - trains the model on the given dataset and tests it on another or on the same dataset
    - [Adversarial Domain Adaptation](trainings/train_ADA.py) - trains the model on GTA5 and tests Cityscapes using Adversarial Domain Adaptation
    - [Fourier Domain Adaptation](trainings/train_FDA.py) - trains the model on GTA5 and tests Cityscapes using Fourier Domain Adaptation
    - [Semi-supervised Fourier Domain Adaptation](trainings/train_SSL_FDA.py.py) - trains the model on GTA5 and tests Cityscapes using Semi-supervised Fourier Domain Adaptation
    - [Validation](trainings/val.py) - validates the model on the given dataset
  - `utils/`: contains utility functions
    - [General utilities](utils/utils.py) - contains utility functions
    - [FDA related utilities](utils/fda.py) - contains utility functions for Fourier Domain Adaptation
    - [Image transformations](utils/transforms.py) - contains image transformations
  - main.py: main script that runs the training and validation


## Information
  - the two datasets are not included in the repository
  - We tested two sets of hyperparameters:
    - config1: optimizer=adam, batch_size=4
    - config2: optimizer=sgd, batch_size=4

## Useful links
-  [Colabs env](https://colab.research.google.com/drive/1TDjhWjOQwZ8ToXjDGRF43G7Qk590C7jP)
- [Docs](https://drive.google.com/drive/folders/1_a1j7FWd2zgzU6ZLaQybO9f02mJM_Uyo)



## Steps and Results
1. **TESTING REAL-TIME SEMANTIC SEGMENTATION**
  1.A - Define the upper bound for the domain adaptation phase.

      ```bash
      action = 'train_simple_cityscapes'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      save_model_path = 'AML_Semantic_DA_project/checkpoints/'
      num_epochs = 50
      num_workers = 4
      save_keyword = 'City_Simple_test_1'
      batchsize=8
      lr=0.01
      optimizer = 'sgd'
      batch_size = 8

      python AML_Semantic_DA_project/main.py --action {action} --pretrain_path {pretrain_path} --num_epochs 50 --num_workers 4 --save_model_path {save_model_path} --learning_rate {lr} --optimizer {optimizer} --batch_size {batch_size}

      ```

    | Config  | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |---------|----------------|------------|-----------------------------|
    | config1 |      78.2      |            |                             |
    | config2 |                |            |                             |



  1.B - Train on the synthetic dataset.

      ```bash

      ```

      | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |----------------|------------|-----------------------------|


  1.C - Evaluate the domain shift.

      Test the model trained at step B on the cityscapes val set.

      ```bash

      ```

      | Accuracy _(%)_ | mIoU _(%)_ |
      |----------------|------------|

  1.D - Try to perform some augmentation techniques during training of STDC on GTA. Set the probability to perform augmentation to 0.5.

      ```bash

      ```

      | Augmentation        | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------------------|----------------|------------|-----------------------------|

2. **IMPLEMENTING UNSUPERVISED ADVERSARIAL DOMAIN ADAPTATION** - Perform adversarial training with labelled synthetic data (source) and unlabelled real-word data (target).

    ```bash

    ```

    | Augmentation        | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |---------------------|----------------|------------|-----------------------------|



3. **IMPROVEMENTS - Image-to-image translation to improve domain adaptation**
    You can implement a fast image-to-image translation algorithm like FDA to improve the overall domain adaptation performances. Test it and compare to step 3 results.

      ```bash

      ```

      | beta | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |------|----------------|------------|-----------------------------|

  3.B - Evaluate the performance of the Segmentation Network adapted with MBT.
      

      ```bash

      ```

      | Accuracy _(%)_ | mIoU _(%)_ |
      |----------------|------------|


  3.C - Self-learning with pseudo-labels.


      Pseudo label generation
      ```bash

      ```

      Training
      
      ```bash

      ```
      | beta | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |------|----------------|------------|-----------------------------|


## Results

Common parameters:
- Base model: STDC
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

saveFile: cityscapes_adam_noaug
date: 10/04/2024
average time: 4:39
precision per pixel for test: 0.787 
mIoU for validation: 0.444 
mIoU per class: [0.95377805 0.67910142 0.82291703 0.24003275 0.21085642 0.24693448 
 0.13421382 0.358522   0.84801198 0.44508914 0.88199773 0.48100367 
 0.20703901 0.82966513 0.15454098 0.22780025 0.13120532 0.1293023 
 0.46101077]

saveFile: cityscapes_adam_aug
date: 
average time:

saveFile: cityscapes_sgd_noaug
date: 
average time:

saveFile: cityscapes_sgd_aug
date: 
average time:

 
- **training & validation on GTA5:**

saveFile: gta5_adam_noaug
date: 10/04/2024
average time: 
precision per pixel for test: 0.782 
mIoU for validation: 0.495 
mIoU per class: [0.96391841 0.70049577 0.7932551  0.55256778 0.24198456 0.35509488 
 0.32561131 0.240652   0.74605463 0.6721074  0.94207433 0.19218172 
 0.16312223 0.64258123 0.53373726 0.44477375 0.61858683 0.24296685 
 0.02396857]

!!! Running validation again
saveFile: gta5_adam_aug
date: 
average time: 

saveFile: gta5_sgd_noaug
date:
average time:
precision per pixel for test: 0.808 
mIoU for validation: 0.640 
mIoU per class: [0.9819757  0.78998446 0.85908591 0.66922654 0.41422396 0.44041994 
 0.43625819 0.42785542 0.79300035 0.72702806 0.94575292 0.42805126 
 0.46199698 0.86912983 0.79643663 0.82218002 0.80119309 0.49816757 
 1.        ]

saveFile: gta5_sgd_aug
date:
average time:
precision per pixel for test: 0.806 
mIoU for validation: 0.625 
mIoU per class: [0.98028663 0.77518529 0.85500451 0.66187531 0.39472562 0.42676516 
 0.42073286 0.42360109 0.78502997 0.71265971 0.94359728 0.41868574 
 0.35928971 0.85548173 0.78355699 0.77141353 0.78301049 0.51820673 
 0.        ]

- **Domain shift evaluation GTA5>Cityscapes:**

saveFile:
date:
average time: 
precision per pixel for test: 0.521
mIoU for validation: 0.115
mIoU per class: [7.61754694e-01 1.71559990e-02 2.95094636e-01 1.65790444e-05
 0.00000000e+00 5.45316307e-03 1.28892443e-03 0.00000000e+00
 2.86752054e-01 4.46510915e-02 4.82411080e-01 0.00000000e+00
 0.00000000e+00 2.91239577e-01 1.42408418e-03 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00]

saveFile:
date:
average time: 

- **ADA without data aug:**

saveFile:
date:
average time: 
precision per pixel for test: 0.594
mIoU for validation: 0.193
mIoU per class: [6.81801228e-01 1.07932154e-01 6.66776069e-01 5.99369935e-02
 6.49155477e-02 1.30033783e-01 5.31078158e-02 6.19663108e-03
 6.49269879e-01 9.55536141e-02 6.63765725e-01 7.46360803e-02
 8.57146883e-03 3.10703607e-01 4.97428665e-02 1.42098133e-02
 4.24880613e-03 1.70739188e-02 2.84853135e-04]

- **ADA with data aug:**

saveFile:
date: 26/05/2024
average time: 
precision per pixel for test: 0.672
mIoU for validation: 0.201
mIoU per class: [8.55649679e-01 8.32756948e-02 6.50656282e-01 7.27817810e-02
 3.19655737e-03 8.41106085e-02 2.09969896e-02 2.44836400e-04
 6.60987629e-01 1.58780603e-01 6.76856284e-01 1.07935027e-02
 0.00000000e+00 5.04739004e-01 3.91791383e-02 2.96842603e-04
 1.00073053e-05 1.37119106e-03 0.00000000e+00]


 - **FDA with beta=0.01**


precision per pixel for test: 0.676
mIoU for validation: 0.215
mIoU per class: [8.64950383e-01 2.58546314e-01 6.79501588e-01 9.01757959e-02
 1.67579924e-02 1.39674604e-01 9.53932518e-03 1.79009915e-03
 6.35830267e-01 8.65353361e-02 6.97690896e-01 3.05773942e-02
 7.89924694e-03 5.29282088e-01 3.60621871e-02 8.49216333e-06
 0.00000000e+00 8.28850874e-03 8.61225770e-06]

 - **FDA with beta=0.05**
 - **FDA with beta=0.01**  
