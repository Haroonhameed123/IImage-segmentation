# Image-segmentation
Class Assigment use two models Mask R cnn and Segnet
Segmentation Projects

This repo contains two image segmentation models:

Mask R-CNN – Detects and segments person, cat, sports ball, and book using Detectron2. Outputs include masks, bounding boxes, and overlay images.

SegNet (VGG16) – Performs semantic segmentation with skip connections. Outputs include predicted masks, overlays, and training plots for mIoU and F1 score.

Install
# For Mask R-CNN
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/detectron2.git

# For SegNet
pip install segmentation_models_pytorch albumentations pycocotools opencv-python matplotlib tqdm

Usage

Update dataset paths in the scripts.

Run training scripts:

python train_maskrcnn.py
python train_segnet.py


Run inference to save predicted masks and overlay images.

Outputs

Mask R-CNN: instance masks with bounding boxes

SegNet: semantic masks, overlay images, training plots
