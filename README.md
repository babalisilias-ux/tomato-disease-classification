# Tomato Leaf Disease Classification

A deep-learning image classifier that identifies tomato leaf diseases from photographs, using transfer learning on MobileNetV2. Achieves ~90% validation accuracy across 10 classes.

## Overview

This project classifies tomato leaves as healthy or as having one of 9 common diseases, using the PlantVillage dataset. It uses transfer learning: a MobileNetV2 model pre-trained on ImageNet is reused as a feature extractor, with a small classifier trained on top.

## Results

| Metric | Score |
|---|---|
| Validation accuracy | 89.8% |
| Classes | 10 (healthy + 9 diseases) |
| Validation images | 3,632 |

The model shows no overfitting. Most errors occur between visually similar diseases (e.g. Early blight vs. Septoria leaf spot).
![Training curves](training_curves.png)
![Confusion matrix](confusion_matrix.png)
## Approach

1. Compute image dataset from PlantVillage (tomato classes only).
2. Resize images to 224x224, apply data augmentation (flip, rotation, zoom).
3. Use frozen MobileNetV2 (ImageNet weights) + dense softmax classifier.
4. Train with 80/20 split, Adam optimiser, 5 epochs.
5. Evaluate with training curves, confusion matrix, and classification report.

## How to run

Designed for Google Colab (free GPU). Open the script in a Colab notebook, set Runtime to GPU, and run top to bottom.

## Tech stack

Python, TensorFlow / Keras, MobileNetV2, scikit-learn, Matplotlib, Seaborn.

## Author

Ilias Bampalis - Chemical Engineering, AUTh
