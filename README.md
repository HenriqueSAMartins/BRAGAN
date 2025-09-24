# BRAGAN - Brazilian Roadkill Animal Detection Project

This project uses the BRAGAN dataset, a GAN-augmented collection of images for detecting roadkill animals on Brazilian highways.

## Dataset Link
You can access the full dataset [here](https://data.mendeley.com/datasets/ck88dwffgd/2).

## Project Overview
BRAGAN is an augmented dataset aimed at improving object detection models for wildlife monitoring, specifically for animals commonly involved in roadkill incidents. The dataset was enhanced using Generative Adversarial Networks (GANs) to generate additional synthetic images, increasing the diversity and volume of data available for training models.

## Highlights

- **9,238 images** total: **1,823** real + **7,300+** classic augmentations + **115** carefully-curated WGAN-GP synthetic images.  
- **5 classes:** tapir, jaguarundi, maned wolf, puma, giant anteater.  
- **YOLO-ready** (YOLO format labels, 80/20 train/val split).  
- **End-to-end pipeline:** preprocessing → GAN training → qualitative filtering → YOLO evaluation.

## 📦 Project Structure

```
BRAGAN/
├─ preprocessing_images/              # Preprocessing: filtering, mirror padding, bbox cropping and resizing
│  ├─ crop_trick/                     # Routines for cropping + mirror padding to build square canvas from YOLO labels
│  └─ data_aug/                       # Classical augmentations (flip, shift, rotate, brightness/contrast/saturation, etc.)
├─ GAN_training/                      # Class-specific GAN training (e.g., WGAN-GP) and utilities
│  ├─ inference/                      # Sample generation (generator inference), saving and curation/selection
│  └─ WGAN-GP=256/                    # WGAN-GP configuration and code for 256×256 resolution (architecture, loops, checkpoints)
└─ YOLO_evaluation/
   ├─ inference/                      # Scripts/notebooks for running inference and collecting metrics
   └─ YOLO_training/                  # Training configs/helpers (data.yaml, hypers, scenarios 1/2/3) for YOLOv5/v8/v11
```

> The dataset itself (**images/** and **labels/**) is **not** stored in this repo. Download it from the link above.

## 🧪 Scenarios
- Scenario 1: real images only
- Scenario 2: Scenario 1 + classical augmentations
- Scenario 3: Scenario 2 + curated WGAN-GP images

## Preprocessing Pipeline
- Lateral pose filtering (reduce silhouette variance)
- Mirror padding + light blur (square canvas)
- Bbox crop from YOLO labels (proportional)
- Resize to 256×256
- Classical augmentations

## GAN Training (WGAN-GP)
- Latent: 128-D z ~ N(0,1)
- Resolution: 256×256 RGB
- Optimizer: Adam (lr=2e-4, β1=0.5, β2=0.999)
- ~800 epochs, 5 critic steps per generator step
- GP λ = 10 (Lipschitz constraint)

## 🔗 YOLO Trained Weights
Final YOLO training weights are available at the following link:
👉 [Google Drive — YOLO Weights](https://drive.google.com/drive/folders/1aVDQh6e_sYjavH-xqvzEUv2eZVexOEBT?usp=sharing).


## 📑 Citing BRAGAN

If you use BRAGAN (data, code, or results), please cite:
```
@dataset{bragan_2025,
  author    = {Souza de Abreu Martins, Henrique and Souto Ferrante, Gabriel and Ipolito Meneguette, Rodolfo},
  title     = {BRAGAN: a GAN-augmented dataset of Brazilian roadkill animals for object detection},
  year      = {2025},
  publisher = {Mendeley Data},
  version   = {V2},
  doi       = {10.17632/ck88dwffgd.2},
  url       = {https://doi.org/10.17632/ck88dwffgd.2}
}
```

## 📬 Contact
Henrique Souza de Abreu Martins — (USP/ICMC)
Gabriel Souto Ferrante — (UFSCar)
Rodolfo I. Meneguette — (ICMC/USP)
