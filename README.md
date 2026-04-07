# Image Classification using Transfer Learning

Dog vs Cat image classifier using transfer learning with VGG16 in Keras, including training, validation, and batch prediction pipeline.

## Overview

This project implements an image classification system that distinguishes between dogs and cats using transfer learning with a pretrained **VGG16** model. It demonstrates how deep convolutional neural networks can be reused for efficient training on limited datasets.

---

## Features

- Transfer learning using pretrained CNN  
- Efficient training with frozen convolutional layers  
- Image preprocessing pipeline  
- Validation-based evaluation  
- Batch prediction on test dataset  
- CSV output generation  

---

## Concepts Used

- Convolutional Neural Networks (CNNs)  
- Transfer Learning  
- Feature Extraction  
- Image Preprocessing  
- Model Evaluation  

---

## Tech Stack

- Keras  
- NumPy  
- Pillow  
- h5py  

---

## Project Structure
project/
│── data/
│ ├── train/
│ │ ├── dogs/
│ │ └── cats/
│ ├── validation/
│ │ ├── dogs/
│ │ └── cats/
│ └── test/
│ └── test/
│
│── img_clf.py
│── vgg_bn.py
│── cats_n_dogs.ipynb
│── cats_n_dogs_BN.ipynb
│
│── prediction.csv


---

## How to Run

1. Arrange dataset in the required folder structure  

2. Install dependencies:
pip install keras numpy pillow h5py


3. Run the script:


python img_clf.py


4. Predictions will be saved in:


prediction.csv


---

## Results

- Training Accuracy: ~85–90% (depends on dataset)  
- Validation Accuracy: ~80–85%  

---

## Limitations

- Limited to binary classification (cats vs dogs)  
- Performance depends heavily on dataset quality  
- No real-time or deployment pipeline  

---

## Future Improvements

- Replace VGG16 with ResNet or EfficientNet  
- Add data augmentation  
- Visualize training (accuracy/loss graphs)  
- Add confusion matrix  
- Deploy as a web app (Flask or FastAPI)  
- Implement Grad-CAM for model interpretability  

---

## Learning Outcomes

- Practical understanding of transfer learning  
- Working with CNN architectures  
- Building an end-to-end ML pipeline  
- Evaluating model performance  

---

## Author

Jaiveer Singh  

---

## Note

This is a foundational computer vision project. To make it stronger for interviews or a resume, extend it with deployment, improved architectures, and proper evaluation metrics.
