# Action Recognition using Vision Transformers
This repository contains the project for the EEEM068 module

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#gettingstarted)
- [Installation](#installation)
- [Dataset](#dataset)
- [License](#license)
## Introduction
This project explores action recognition in videos using Vision Transformers, with a focus on the TimeSformer architecture. By evaluating various frame sampling strategies, augmentation techniques, and model configurations on the HMDB_simp dataset, the study achieves a top accuracy of 90.3% and demonstrates the effectiveness of transformer-based approaches for capturing spatiotemporal patterns in video data.


## Getting Started
### Prerequisites
- Python 3.10
- Jupyter Notebook

### Installation
1. Clone this repository: git clone https://github.com/Elisa-tea/EEEM068.git
2. Install dependencies:
```
pip install \
  torch torchvision \
  albumentations albucore \
  scikit-learn matplotlib pandas tqdm ipykernel \
  fastapi uvicorn \
  transformers datasets evaluate \
  gradio wandb accelerate torchmetrics \
  simsimd stringzilla tf-keras
```
### Run the program
#### 1. train.py
for example, for fixed-step sampling and a clip length of 8, run the following command in the terminal:
```
python train.py --sampler fixed_step --frame_step 8 --clip_length 8 --train_batch_size 4 --lr 0.00001 --weight_decay 0.095 --use_augmentations(optional)
```
#### 2. GradCAM2.ipynb
## Dataset
The HMDB_simp dataset includes 1,250 videos, with 50 videos per category. Each subfolder of the dataset corresponds
to a different action category. The dataset includes 1,250 videos, with 50 videos per category.
The dataset used in this project is HMDB_simp_cleaned, a cleaned version of HMDB_simp (without duplicated frames)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
## Acknowledgments
We would like to thank all group members for their contributions to this project:
- [Elizaveta Andrushkevich](https://github.com/Elisa-tea)
- [Kavitha Appulingam](https://github.com/Kavithaaa23)
- [Yiwen Chan](https://github.com/v41827)
- [Artem Karamyshev](https://github.com/ArtemKar123)
