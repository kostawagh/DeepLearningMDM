# CIFAR-10 Image Classification Using Transfer Learning and Feature Fusion

## Overview

This project focuses on classifying images from the CIFAR-10 dataset using a transfer learning approach with a VGG16 model. The goal is to evaluate the model's performance and explore areas for improvement by comparing it with a research paper that combines features from manual methods (HOG, pixel intensities) and deep learning models (VGG16, Inception ResNet v2) to improve classification accuracy.

## Objective

The main objective is to:
- Fine-tune a pre-trained VGG16 model on the CIFAR-10 dataset.
- Evaluate the performance using accuracy, precision, recall, and F1-score.
- Compare the results with those reported in the research paper "CIFAR-10 Image Classification Using Feature Ensembles" by Felipe O. Giuste and Juan C. Vizcarra.

## Dataset

The dataset used in this project is the **CIFAR-10** dataset, which consists of 60,000 32x32 color images in 10 classes (6,000 images per class). The dataset is divided into:
- 50,000 training images
- 10,000 test images

Classes:  
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Methodology

1. **Preprocessing**:
   - The CIFAR-10 dataset is preprocessed by normalizing the pixel values to a range of [0, 1].
   - Data augmentation (horizontal flipping) was applied to improve model generalization.

2. **Model Architecture**:
   - The model uses **VGG16** pre-trained on ImageNet and fine-tuned on CIFAR-10.
   - The last layers of the VGG16 model were unfrozen for fine-tuning.
   
3. **Training**:
   - The model was trained for 20 epochs with early stopping to prevent overfitting.
   - A fully connected neural network (FCNN) was used for training the fine-tuned VGG16 features.

## Results

- **Our Model’s Accuracy**: 75.92%
- **Precision, Recall, F1-Score**: 
  - Precision: 75.75%
  - Recall: 75.92%
  - F1-Score: 75.59%

## Comparison with Research Paper

The research paper "CIFAR-10 Image Classification Using Feature Ensembles" reported the following results for their ensemble models:
- **Ensemble (TL-VGG + HOG + Pixel Intensities):** 85%
- **Ensemble (TL-VGG + TL-Inception ResNet v2):** 91.12%
- **All 5 Features Combined (Top 1000 PCA components):** 94.6%

### Comparison with Our Model:
- Our model achieved an accuracy of 75.92%, which is lower than the reported ensemble approaches (85% and 94.6%).
- The difference in performance may be due to the use of a single transfer learning model (VGG16) in our approach compared to the ensemble of multiple models and feature types in the paper.

## Areas for Improvement

### Weaknesses:
- **Accuracy Gap**: Our model's accuracy is lower than the paper’s results.
- **Limited Features**: Our model used only VGG16 and did not incorporate other feature extractors (e.g., HOG, pixel intensities).
- **Lack of Feature Fusion**: Unlike the paper, PCA was not applied to fuse features from multiple methods.

### Suggested Improvements:
- **Incorporating More Features**: Adding features like HOG and pixel intensities could capture complementary information.
- **Feature Fusion**: Applying PCA or other techniques could improve feature diversity and enhance performance.
- **Ensemble Models**: Combining the results of multiple fine-tuned models (e.g., VGG16 and Inception ResNet v2) could increase accuracy.
- **More Data Augmentation**: Implementing additional augmentation techniques (rotations, zooms, shifts) could help improve generalization.

## Conclusion

Although our model achieved a respectable accuracy of 75.92%, it still falls short compared to the ensemble models in the research paper. By incorporating additional features, applying advanced feature fusion, and experimenting with ensemble models, we can improve the model's performance on the CIFAR-10 dataset.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Scikit-learn

