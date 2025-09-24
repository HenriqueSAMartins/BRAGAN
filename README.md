# BRAGAN - Brazilian Roadkill Animal Detection Project

This project uses the BRAGAN dataset, a GAN-augmented collection of images for detecting roadkill animals on Brazilian highways.

## Dataset Link
You can access the full dataset [here](https://data.mendeley.com/datasets/ck88dwffgd/2).

## Project Overview
BRAGAN is an augmented dataset aimed at improving object detection models for wildlife monitoring, specifically for animals commonly involved in roadkill incidents. The dataset was enhanced using Generative Adversarial Networks (GANs) to generate additional synthetic images, increasing the diversity and volume of data available for training models.

## ✨ Highlights

- **9,238 images** total: **1,823** real + **7,300+** classic augmentations + **115** carefully-curated WGAN-GP synthetic images.  
- **5 classes:** tapir, jaguarundi, maned wolf, puma, giant anteater.  
- **YOLO-ready** (YOLO format labels, 80/20 train/val split).  
- **End-to-end pipeline:** preprocessing → GAN training → qualitative filtering → YOLO evaluation.

## 📦 Project Structure

BRAGAN/
├─ preprocessing_images/ # Scripts/notebooks for filtering, padding, cropping, resizing, and classical aug.
├─ GAN_training/ # Class-specific GAN setups (e.g., WGAN-GP) and training utilities
└─ YOLO_evaluation/
    └─ YOLO/ # Training/evaluation configs & helpers for YOLOv5/v8/v11

> The dataset itself (**images/** and **labels/**) is **not** stored in this repo. Download it from the link above.

