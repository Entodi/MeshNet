# End-to-end learning for brain tissue segmentation

This repository contains Torch implementation of MeshNet architecture. MeshNet is volumetric convolutional neural network based on dilated kernels [1] for image segmentation. Model has been trained for brain tissue segmentation from imperfect labeling obtained using FreeSurfer automatic approach. The repository also contains weights of trained model with volumetric dropout.

# Details

The repository has following structure:

- **./models/**
  Contains code for Deep Neural network architectures
  - **vdp_model.lua**
    MeshNet model with volumetric dropout
  - **nodp_model.lua**
    MeshNet model without volumetric dropout
- **./saved_models/**
  Contains saved weights and csv with train and validation loss during training  
  - **./model_Mon_Jul_10_16:43:55_2017/**  
  best weights and loss logs for **../models/vdp_model.lua**
- **train.lua**  
  Torch Lua code for training models
- **metrics.lua**  
  Torch Lua code for calculating F1 (equivalent to DICE score) and AVD metrics and for saving prediction.
- **utils.lua**  
  Torch Lua code for utility functions.
- **prepare_data.lua**  
  Example code for preparing data from numpy format to torch format. Maps intensity to unit interval.
- **train.sh**  
  Example bash script for model training. The script has been used to train saved model.
- **metrics.sh**  
  Example bash script for calculating metrics and saving prediction
- **mklabels.sh**  
  Bash script to prepare data and labels to numpy format from Human Connectome Project [3]. (**IMPORTANT: labels have been fixed after expert review**)
- **train_fold.txt**  
  Training fold with 20 subjects
- **valid_fold.txt**  
  Validation fold with 2 subjects

Model has been trained on 20 subjects T1 3T MRI images with slice thickness 1mm x 1mm x 1mm (256 x 256 x 256) from Human Connectome Project [3] and validated on 2 subjects during training.
More details about the training process are published at IJCNN 2017 and described in a more up to date paper [2]. **IMPORTANT: model on github uses Volumetric Dropout instead of 1D Dropout (due to significant improvements). One epoch consists of 2048 subvolumes with size 68 x 68 x 68 and validated on same amount of subvolumes. Model is 219 epoch old.**

Code is written on Lua using Torch deep learning library (http://torch.ch/).
Additional packages are required: torch-randomkit (https://github.com/deepmind/torch-randomkit), npy4th (https://github.com/htwaijry/npy4th), torch-dataframe (https://github.com/AlexMili/torch-dataframe), csvigo (https://github.com/clementfarabet/lua---csv).

Model has been trained using NVIDIA Titan X (Pascal) with 12 GB. Model is using 9817 MB of GPU memory during training with batch size 1. Train time is about 3-4 days.

# Result on subject **105216**
| T1 MRI  | FreeSurfer | MeshNet |
|---|---|---|
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_t1.gif?raw=true)  |  ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_fs.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/axial_219.gif?raw=true)   |
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_t1.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_fs.gif?raw=true)   | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/sagittal_219.gif?raw=true)   |
| ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_t1.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_fs.gif?raw=true)  | ![Alt Text](https://github.com/Entodi/MeshNet/blob/master/gif/coronal_219.gif?raw=true)  |

# References
[1] https://arxiv.org/abs/1511.07122 Multi-Scale Context Aggregation by Dilated Convolutions. *Fisher Yu, Vladlen Koltun*  
[2] https://arxiv.org/abs/1612.00940 End-to-end learning of brain tissue segmentation from imperfect labeling. *Alex Fedorov, Jeremy Johnson, Eswar Damaraju, Alexei Ozerin, Vince D. Calhoun, Sergey M. Plis*  
[3] http://www.humanconnectomeproject.org/ Human Connectome Project

# Questions

You can ask any questions about implementation and training by sending message to **afedorov@mrn.org**.
