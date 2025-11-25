# Image-segmentation
Class Assigment use two models Mask R cnn and Segnet
Semantic Segmentation & Instance Segmentation Projects

This repository contains implementations of two deep learning models for image segmentation:

Mask R-CNN – Instance Segmentation

SegNet (VGG16) – Semantic Segmentation

1. Mask R-CNN

Detects and segments person, cat, sports ball, book

Uses Detectron2 library with pre-trained COCO weights

Dataset filtered to target classes

Trains on custom COCO-style dataset

Outputs: predicted masks, bounding boxes, and overlay visualizations

Evaluates with COCO metrics (AP, AR)

2. SegNet (VGG16)

Semantic segmentation with skip connections

Trains with Dice + Weighted CrossEntropy Loss

Data augmentation using Albumentations

Evaluation metrics: mIoU, F1 score, precision, recall

Outputs: predicted masks and overlays

Setup
# PyTorch & dependencies
pip install torch torchvision torchaudio
pip install segmentation_models_pytorch albumentations pycocotools opencv-python matplotlib tqdm
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.1/index.html

Usage

Prepare COCO-style dataset for both tasks.

Update paths in the scripts (TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR).

Train Mask R-CNN:

python train_maskrcnn.py


Train SegNet:

python train_segnet.py


Run inference and save predictions.

Outputs

Mask R-CNN: segmented instances with bounding boxes

SegNet: semantic masks and overlay visualizations

Training plots: loss curves, mIoU, F1 score

Confusion matrix for SegNet
