# Image-segmentation
Class Assigment use two models Mask R cnn and Segnet
Semantic Segmentation & Instance Segmentation Projects

This repository contains implementations of two deep learning models for image segmentation:

Mask R-CNN – Instance Segmentation

SegNet (VGG16) – Semantic Segmentation

1. Mask R-CNN

Detects and segments person, cat, sports ball, book

Uses Detectron2 with pre-trained COCO weights

Dataset filtered to target classes

Outputs: predicted masks, bounding boxes, and overlays

Evaluation with COCO metrics

Installation:

# PyTorch + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Detectron2
pip install git+https://github.com/facebookresearch/detectron2.git

2. SegNet (VGG16)

Semantic segmentation with skip connections

Loss: Dice + Weighted CrossEntropy

Data augmentation using Albumentations

Evaluation: mIoU, F1, precision, recall

Outputs: predicted masks, overlay visualizations, loss & metric plots

Installation:

pip install segmentation_models_pytorch albumentations pycocotools opencv-python matplotlib tqdm

Usage

Prepare COCO-style dataset for both tasks.

Update paths in the scripts (TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TEST_IMAGES_DIR).

Train Mask R-CNN:

python train_maskrcnn.py


Train SegNet:

python train_segnet.py


Run inference to save predicted masks and overlay images.

Outputs

Mask R-CNN: segmented instances with bounding boxes

SegNet: semantic masks and overlays

Training plots for loss, mIoU, F1 score

Confusion matrix for SegNet
