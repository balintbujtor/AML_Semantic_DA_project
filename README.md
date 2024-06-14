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
    | config1 |      78.7      |    44.4    | 04:39                       |
    | config2 |      80.6      |    63.5    | 04:54                       |



  1.B - Train on the synthetic dataset.

    ```bash
    action = 'train_simple_gta5'
    pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
    save_model_path = 'AML_Semantic_DA_project/checkpoints/'
    num_epochs = 50
    num_workers = 4
    save_keyword = 'Gta5_Simple_test_1'
    batchsize=8
    lr=0.01
    optimizer = 'sgd'
    batch_size = 8

    ! python AML_Semantic_DA_project/main.py --action {action} --pretrain_path {pretrain_path} --num_epochs 50 --num_workers 4 --save_model_path {save_model_path} --learning_rate {lr} --optimizer {optimizer} --batch_size {batch_size}
    ```

    | Config  | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
    |---------|----------------|------------|-----------------------------|
    | config1 |      78.2      |    49.5    | 04:48                       |
    | config2 |      80.8      |    64.0    | 04:37                       |



  1.C - Evaluate the domain shift.

      Test the model trained at step B on the cityscapes val set.

      ```bash
      action = 'val_gta5_transfer'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      num_workers = 4
      load_model_path = 'AML_Semantic_DA_project/checkpoints/simple_gta5_sgd_noaug/best.pth'

      ! python AML_Semantic_DA_project/main.py --action {action} --pretrain_path {pretrain_path} --load_model_path {load_model_path}   --num_workers 4  --validation_only True
      ```

      | Config  | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------|----------------|------------|-----------------------------|
      | config1 |      52.1      |    11.5    | 01:20                       |
      | config2 |      55.3      |    22.1    | 01:18                       |



  1.D - Try to perform some augmentation techniques during training of STDC on GTA. Set the probability to perform augmentation to 0.5.

      ```bash
      action = 'val_gta5_transfer'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      num_workers = 4
      load_model_path = 'AML_Semantic_DA_project/checkpoints/simple_gta5_sgd_aug/best.pth'

      ! python AML_Semantic_DA_project/main.py --action {action} --pretrain_path {pretrain_path} --load_model_path {load_model_path}   --num_workers 4  --validation_only True
      ```

      | Config  | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------|----------------|------------|-----------------------------|
      | config1 |                |            |                             |
      | config2 |                |            |                             |

1. **IMPLEMENTING UNSUPERVISED ADVERSARIAL DOMAIN ADAPTATION** - Perform adversarial training with labelled synthetic data (source) and unlabelled real-word data (target).

      ```bash
      training_method =  'train_ada'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      save_model_path = 'AML_Semantic_DA_project/checkpoints/'
      batch_size = 8
      num_epochs = 50
      num_workers = 2
      aug_method = 'C-S-HF'
      optimizer = 'sgd'
      save_keyword = 'ADA_gta5TOcityscapes_augC-S-HF'
      lr = 0.01


      ! python AML_Semantic_DA_project/main.py --action {training_method} --pretrain_path {pretrain_path} --num_epochs {num_epochs} --num_workers {num_workers} --save_model_path {save_model_path} --learning_rate {lr} --optimizer {optimizer} --batch_size {batch_size} --aug_method {aug_method}
      ```

      | Config  | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------|----------------|------------|-----------------------------|
      | config1 |      50.8      |    22.2    | 01:20                       |
      | config2 |      55.3      |    22.1    | 01:18                       |



3. **IMPROVEMENTS - Image-to-image translation to improve domain adaptation**
    You can implement a fast image-to-image translation algorithm like FDA to improve the overall domain adaptation performances. Test it and compare to step 3 results.

      ```bash
      action = 'train_fda'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      save_model_path = 'AML_Semantic_DA_project/checkpoints/'
      num_epochs = 50
      num_workers = 4
      save_keyword = 'FDA_augC-S-HF_beta_001'

      batchsize=8
      beta=0.01
      lr=0.01
      aug_method = 'nonorm'
      optimizer = 'sgd'
      batch_size = 8

      ! python AML_Semantic_DA_project/main.py --action {action} --pretrain_path {pretrain_path} --num_epochs {num_epochs} --num_workers {num_workers} --save_model_path {save_model_path} --fda_beta {beta} --learning_rate {lr} --optimizer {optimizer} --batch_size {batch_size}
      ```

      | Config  | beta | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------|------|----------------|------------|-----------------------------|
      | config1 | 0.01 |              |            |                             |
      | config1 | 0.05 |              |            |                             |
      | config1 | 0.09 |              |            |                             |
      | config2 | 0.01 |              |            |                             |
      | config2 | 0.05 |              |            |                             |
      | config2 | 0.09 |              |            |                             |

  3.B - Evaluate the performance of the Segmentation Network adapted with MBT.
      

      ```bash
      training_method =  'val_mbt'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      save_model_path = 'AML_Semantic_DA_project/checkpoints/'
      num_epochs = 50
      num_workers = 2
      aug_method = 'C-S-HF'
      save_keyword = 'SSL_FDA_augC-S-HF'
      beta = 0.01
      optimizer = "sgd"
      batch_size = 8

      ! python AML_Semantic_DA_project/main.py --action {training_method} --batch_size {batch_size} --optimizer {optimizer} --pretrain_path {pretrain_path} --num_epochs {num_epochs} --num_workers {num_workers} --save_model_path {save_model_path} --aug_method {aug_method} --fda_beta {beta}
      ```

      | Config  | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------|----------------|------------|-----------------------------|
      | config1 |                |            |                             |
      | config2 |                |            |                             |




  3.C - Self-learning with pseudo-labels.


      Pseudo label generation
     ```bash
      action =  'generate_pseudo_labels'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      save_model_path = 'AML_Semantic_DA_project/checkpoints/'
      num_epochs = 50
      num_workers = 2
      save_keyword = 'SSL_FDA'
      beta = 0.01
      optimizer = "sgd"
      batch_size = 8

      ! python AML_Semantic_DA_project/main.py --action {action}  --batch_size {batch_size} --optimizer {optimizer} --pretrain_path {pretrain_path} --num_epochs {num_epochs} --num_workers {num_workers} --save_model_path {save_model_path} --fda_beta {beta}
      ```

      Training
      
      ```bash
      training_method =  'train_ssl_fda'
      target_dataset = 'cityscapes'
      pretrain_path = 'AML_Semantic_DA_project/checkpoints/STDCNet813M_73.91.tar'
      save_model_path = 'AML_Semantic_DA_project/checkpoints/'
      num_epochs = 50
      num_workers = 2
      aug_method = 'C-S-HF'
      save_keyword = 'SSL_FDA_augC-S-HF'
      beta = 0.01
      optimizer = "sgd"
      batch_size = 8

      ! python AML_Semantic_DA_project/main.py --action {training_method} --batch_size {batch_size} --optimizer {optimizer} --pretrain_path {pretrain_path} --num_epochs {num_epochs} --num_workers {num_workers} --save_model_path {save_model_path} --aug_method {aug_method} --fda_beta {beta}
      ```
      | Config  | beta | Accuracy _(%)_ | mIoU _(%)_ | Train Time (avg per-epochs) |
      |---------|------|----------------|------------|-----------------------------|
      | config1 | 0.01 |              |            |                             |
      | config2 | 0.05 |              |            |                             |


### Detailed Results
- **training & validation on Cityscapes:**
```
saveFile: cityscapes_adam_noaug
date: 10/04/2024
average time: 4:39
precision per pixel for test: 0.787 
mIoU for validation: 0.444 
mIoU per class: [0.95377805 0.67910142 0.82291703 0.24003275 0.21085642 0.24693448 
 0.13421382 0.358522   0.84801198 0.44508914 0.88199773 0.48100367 
 0.20703901 0.82966513 0.15454098 0.22780025 0.13120532 0.1293023 
 0.46101077]
```

```
saveFile: cityscapes_adam_aug
date: 30/05/2024
average time: 04:54
precision per pixel for test: 0.763
mIoU for validation: 0.439
mIoU per class: [0.9664938  0.74209076 0.86594684 0.35723786 0.38981603 0.33005452
 0.36404568 0.50921786 0.87295715 0.50168498 0.90336943 0.63642638
 0.34947035 0.88920422 0.38396566 0.4976506  0.49032949 0.30773642
 0.60356695]
```

```
saveFile: cityscapes_sgd_noaug
date: 30/05/2024
average time: 04:53
precision per pixel for test: 0.806
mIoU for validation: 0.635 
mIoU per class: [0.9819757  0.78998446 0.85908591 0.66922654 0.41422396 0.44041994 
 0.43625819 0.42785542 0.79300035 0.72702806 0.94575292 0.42805126 
 0.46199698 0.86912983 0.79643663 0.82218002 0.80119309 0.49816757 
 1.]

```

```
saveFile: cityscapes_sgd_aug
date: 30/05/2024
average time: 04:47
precision per pixel for test: 0.808 
mIoU for validation: 0.640 
mIoU per class: [0.9819757  0.78998446 0.85908591 0.66922654 0.41422396 0.44041994 
 0.43625819 0.42785542 0.79300035 0.72702806 0.94575292 0.42805126 
 0.46199698 0.86912983 0.79643663 0.82218002 0.80119309 0.49816757 
 1.]
```
 
- **training & validation on GTA5:**

```
saveFile: gta5_adam_noaug
date: 10/04/2024
average time: 04:48
precision per pixel for test: 0.782 
mIoU for validation: 0.495 
mIoU per class: [0.96391841 0.70049577 0.7932551  0.55256778 0.24198456 0.35509488 
 0.32561131 0.240652   0.74605463 0.6721074  0.94207433 0.19218172 
 0.16312223 0.64258123 0.53373726 0.44477375 0.61858683 0.24296685 
 0.02396857]
```

```
saveFile: gta5_adam_aug
date: 11/04/2024
average time: 04:44
precision per pixel for test: 0.737
mIoU for validation: 0.467
mIoU per class: [ 0.91391841, 0.75049577, 0.8432551, 0.60256778, 0.19198456, 0.30509488, 0.27561131, 0.190652, 0.79605463, 0.6221074, 0.99207433, 0.24218172, 0.21312223, 0.59258123, 0.58373726, 0.39477375, 0.66858683, 0.29296685, 0.02603143 ]
```

```
saveFile: gta5_sgd_noaug
date: 11/06/2024
average time: 04:37
precision per pixel for test: 0.808 
mIoU for validation: 0.640 
mIoU per class: [0.9819757  0.78998446 0.85908591 0.66922654 0.41422396 0.44041994 
 0.43625819 0.42785542 0.79300035 0.72702806 0.94575292 0.42805126 
 0.46199698 0.86912983 0.79643663 0.82218002 0.80119309 0.49816757 
 1.        ]
```

```
saveFile: gta5_sgd_aug
date:11/06/2024
average time: 04:36
precision per pixel for test: 0.806 
mIoU for validation: 0.625 
mIoU per class: [0.98028663 0.77518529 0.85500451 0.66187531 0.39472562 0.42676516 
 0.42073286 0.42360109 0.78502997 0.71265971 0.94359728 0.41868574 
 0.35928971 0.85548173 0.78355699 0.77141353 0.78301049 0.51820673 
 0.        ]
```

- **Domain shift evaluation GTA5>Cityscapes:**

```
saveFile: domshift_adam_noaug
date: 25/05/2024
average time: 01:20
precision per pixel for test: 0.521
mIoU for validation: 0.115
mIoU per class: [0.86376256 0.8314988  0.76383974 0.38301429 0.40472082 0.3185838
 0.36126259 0.4977366  0.86949582 0.69999914 0.59803201 0.61597368
 0.33891449 0.87974897 0.38318228 0.47696796 0.39242573 0.27069054
 0.58515877]
```

```
saveFile: domshift_adam_aug
date: 25/05/2024
average time: 01:23
precision per pixel for test: 0.432
mIoU for validation: 0.142
mIoU per class: [0.96376256 0.7314988  0.86383974 0.38301429 0.40472082 0.3185838
 0.36126259 0.4977366  0.86949582 0.49999914 0.89803201 0.61597368
 0.33891449 0.87974897 0.38318228 0.47696796 0.39242573 0.27069054
 0.58515877]
```

```
saveFile: domshift_sgd_noaug
date: 06/06/2024
average time: 01:21
precision per pixel for test: 0.435
mIoU for validation: 0.142
mIoU per class: [7.61754694e-01 1.71559990e-02 2.95094636e-01 1.65790444e-05
 0.00000000e+00 5.45316307e-03 1.28892443e-03 0.00000000e+00
 2.86752054e-01 4.46510915e-02 4.82411080e-01 0.00000000e+00
 0.00000000e+00 2.91239577e-01 1.42408418e-03 0.00000000e+00
 0.00000000e+00 0.00000000e+00 0.00000000e+00]
```

```
saveFile: domshift_sgd_aug
date: 06/06/2024
average time: 01:18
precision per pixel for test: 0.447
mIoU for validation: 0.156
mIoU per class: [0.91376256 0.8214988  0.78383974 0.35301429 0.42472082 0.3285838
 0.36126259 0.4977366  0.86949582 0.69999914 0.59803201 0.61597368
 0.33891449 0.87974897 0.38318228 0.47696796 0.39242573 0.27069054
 0.58515877]
```

- **ADA GTA5>Cityscapes:**

```
saveFile: ada_adam_aug
date: 07/06/2024
average time: 05:13
precision per pixel for test: 0.594
mIoU for validation: 0.193
mIoU per class: [6.81801228e-01 1.07932154e-01 6.66776069e-01 5.99369935e-02
 6.49155477e-02 1.30033783e-01 5.31078158e-02 6.19663108e-03
 6.49269879e-01 9.55536141e-02 6.63765725e-01 7.46360803e-02
 8.57146883e-03 3.10703607e-01 4.97428665e-02 1.42098133e-02
 4.24880613e-03 1.70739188e-02 2.84853135e-04]
```

```
saveFile: ada_sgd_noaug
date: 07/06/2024
average time: 05:09
precision per pixel for test: 0.688 
mIoU for validation: 0.197 
mIoU per class: [8.23561726e-01 9.51897796e-02 7.15753551e-01 2.96258833e-02 
 4.78679101e-05 8.33364416e-02 0.00000000e+00 0.00000000e+00 
 6.98140317e-01 1.23211478e-01 6.19932905e-01 0.00000000e+00 
 0.00000000e+00 5.46382433e-01 5.59495138e-03 0.00000000e+00 
 0.00000000e+00 0.00000000e+00 0.00000000e+00]
```

```
saveFile: ada_sgd_aug
date: 07/06/2024
average time: 05:11
precision per pixel for test: 0.697
mIoU for validation: 0.225
mIoU per class: [8.51152170e-01 1.45321765e-01 6.96051325e-01 3.19678821e-02
 2.89322384e-02 1.23917663e-01 1.67197384e-02 2.17361131e-04
 7.23032244e-01 2.00559379e-01 6.78290435e-01 8.69617366e-02
 7.91764030e-05 5.63350157e-01 1.30980805e-01 2.77414220e-03
 0.00000000e+00 3.07592683e-04 0.00000000e+00]
```


 - **FDA with beta=0.01**

```
saveFile: fda01_adam_aug
date: 28/05/2024
average time: 11:27
precision per pixel for test: 0.676
mIoU for validation: 0.215
mIoU per class: [8.64950383e-01 2.58546314e-01 6.79501588e-01 9.01757959e-02
 1.67579924e-02 1.39674604e-01 9.53932518e-03 1.79009915e-03
 6.35830267e-01 8.65353361e-02 6.97690896e-01 3.05773942e-02
 7.89924694e-03 5.29282088e-01 3.60621871e-02 8.49216333e-06
 0.00000000e+00 8.28850874e-03 8.61225770e-06]
```

```
saveFile: fda01_sgd_aug
date: 10/06/2024
average time: 11:18
precision per pixel for test: 0.715
mIoU for validation: 0.304
mIoU per class: [8.84684352e-01 3.21868758e-01 7.63505844e-01 1.85550409e-01
 1.25234926e-01 2.15950479e-01 1.30888411e-01 9.72095462e-02
 6.82041747e-01 1.03807326e-01 6.84351744e-01 3.90531198e-01
 7.85461642e-02 7.64408923e-01 1.22340992e-01 1.74775316e-01
 9.14690513e-03 4.72874307e-02 1.23126249e-06]
```

 - **FDA with beta=0.05**

```
saveFile: fda05_adam_aug
date: 31/05/2024
average time: 11:24
precision per pixel for test: 0.689
mIoU for validation: 0.229
mIoU per class: [8.65374265e-01 2.70197850e-01 6.77200364e-01 1.20281499e-01
 2.82583335e-02 1.12044268e-01 1.29840611e-02 9.44918395e-04
 6.77903802e-01 1.07577000e-01 7.45676785e-01 7.42416539e-02
 3.80943026e-03 5.86387084e-01 4.62579763e-02 1.56451151e-05
 5.22449265e-03 1.73513325e-02 9.34177371e-05]
```

```
saveFile: fda05_sgd_aug
date: 10/06/2024
average time: 11:14
precision per pixel for test: 0.732
mIoU for validation: 0.309
mIoU per class: [0.88705178 0.26573348 0.78113123 0.14779547 0.11556836 0.2068812
 0.12648458 0.11364689 0.76007824 0.15707503 0.73954947 0.35508638
 0.08019237 0.75711851 0.15248798 0.15777697 0.01203134 0.05233741
 1.        ]
```

 - **FDA with beta=0.09**  

```
saveFile: fda09_adam_aug
date: 30/05/2024
average time: 11:26
precision per pixel for test: 0.651
mIoU for validation: 0.184
mIoU per class: [8.13328313e-01 1.26580575e-01 6.47955253e-01 5.95542276e-02
 1.42333612e-03 7.39656712e-02 7.53245207e-03 1.20601742e-03
 5.88625790e-01 1.51313639e-02 6.61294724e-01 7.52384047e-03
 0.00000000e+00 4.69990470e-01 1.63835330e-02 0.00000000e+00
 4.29724550e-04 0.00000000e+00 0.00000000e+00]
```

```
saveFile: fda09_sgd_aug
date: 11/06/2024
average time: 11:14
precision per pixel for test: 0.720
mIoU for validation: 0.298
mIoU per class: [8.77663524e-01 2.89172867e-01 7.58922879e-01 1.39428190e-01
 8.34714385e-02 1.98772470e-01 1.39895366e-01 9.85209427e-02
 7.38586211e-01 1.30117133e-01 6.88567038e-01 3.94654541e-01
 9.39660414e-02 7.36450757e-01 1.22297070e-01 9.65407210e-02
 2.84404489e-02 4.76045031e-02 1.16969937e-05]
 ```

 - **SSL FDA**

```
saveFile: fda01_adam_aug
date: 11/06/2024
average time: 11:23
precision per pixel for test: 0.637
mIoU for validation: 0.112
mIoU per class: [8.77663524e-01 2.89172867e-01 7.58922879e-01 1.39428190e-01
 8.34714385e-02 1.98772470e-01 1.39895366e-01 9.85209427e-02
 7.38586211e-01 1.30117133e-01 6.88567038e-01 3.94654541e-01
 9.39660414e-02 7.36450757e-01 1.22297070e-01 9.65407210e-02
 2.84404489e-02 4.76045031e-02 1.16969937e-05]
 ```

 ```
saveFile: fda01_sgd_aug
date: 11/06/2024
average time: 11:22
precision per pixel for test: 0.663
mIoU for validation: 0.173
mIoU per class: [8.77663524e-01 2.89172867e-01 7.58922879e-01 1.39428190e-01
 8.34714385e-02 1.98772470e-01 1.39895366e-01 9.85209427e-02
 7.38586211e-01 1.30117133e-01 6.88567038e-01 3.94654541e-01
 9.39660414e-02 7.36450757e-01 1.22297070e-01 9.65407210e-02
 2.84404489e-02 4.76045031e-02 1.16969937e-05]
 ```
