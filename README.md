# Classification of Images from the CIFAR-10 Dataset using ANN (MLP) & CNN  

This project aims to classify images from the **CIFAR-10 dataset** using Artificial Neural Networks (ANNs) and Convolutional Neural Networks (CNNs). It provides a hands-on approach to building, training, and evaluating image classification models with Python libraries such as TensorFlow and Keras.  

## Table of Contents  
- [Objective](#objective)  
- [Dataset](#dataset)  
- [Class Labels](#class-labels)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Setup and Installation](#setup-and-installation)  
- [Usage](#usage)  
- [Results](#results)  
- [Future Enhancements](#future-enhancements)  
- [Contributors](#contributors)  
- [License](#license)  

---

## Objective  
The primary objective of this project is to equip learners with the knowledge and skills to:  
- Preprocess image data for model training.  
- Build and implement ANN (Multilayer Perceptron) and CNN architectures.  
- Train models to classify images into one of ten classes.  
- Evaluate model performance and optimize network architectures for real-world applications.  

## Dataset  
The **CIFAR-10 dataset** is a collection of **60,000 color images** of size **32x32 pixels** across **10 classes**.  
- Training images: **50,000**  
- Test images: **10,000**  

### Class Labels  
The dataset consists of the following classes:  

| Class       | Label |  
|-------------|-------|  
| Airplane    | 0     |  
| Automobile  | 1     |  
| Bird        | 2     |  
| Cat         | 3     |  
| Deer        | 4     |  
| Dog         | 5     |  
| Frog        | 6     |  
| Horse       | 7     |  
| Ship        | 8     |  
| Truck       | 9     |  

---

## Features  
- Implementation of **ANN (MLP)** and **CNN** architectures.  
- Preprocessing image data, including normalization and one-hot encoding.  
- Training and evaluation of models on CIFAR-10.  
- Visualization of loss and accuracy metrics.  

---

## Requirements  
- Python 3.7+  
- TensorFlow 2.0+  
- Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Setup and Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/Mittalkabir/cifar10-classification.git  
   cd cifar10-classification  
2. Create a virtual environment:
python -m venv env  
source env/bin/activate  # On Windows: env\Scripts\activate  

3. Install required libraries:
pip install -r requirements.txt  

# Usage
1.Preprocess the CIFAR-10 dataset:
 python preprocess_data.py  

2.Train the models:
 For ANN:
 python train_ann.py  


3.For CNN:
 python train_cnn.py  


4.Evaluate the models:

 python evaluate_models.py 

### Results  

1. **ANN (MLP) Model**:  
   - **Training Accuracy**: Achieved approximately **85-88%** after 10 epochs.  
   - **Validation Accuracy**: Reached approximately **65-70%** on the test dataset.  
   - The ANN model struggled with more complex patterns in the dataset due to the lack of spatial feature extraction capabilities.  

2. **CNN Model**:  
   - **Training Accuracy**: Achieved around **95-97%** after 10 epochs.  
   - **Validation Accuracy**: Reached approximately **80-85%** on the test dataset.  
   - The CNN model performed significantly better, leveraging its convolutional layers to extract spatial and hierarchical features.  

3. **Loss Metrics**:  
   - ANN validation loss decreased but plateaued early, suggesting limitations in generalization.  
   - CNN validation loss steadily decreased, indicating better learning and feature extraction.  

4. **Predictions**:  
   - The trained CNN model successfully predicted the class of a test image with high confidence.  
   - Example Output:  
     - Input: Test Image of a **Truck**.  
     - Predicted Class: **Truck**.  

5. **Visualizations**:  
   - Loss and accuracy graphs showed consistent improvement for the CNN model, with validation curves closely following training curves, indicating minimal overfitting.  

---

These results highlight the advantages of CNNs for image classification tasks, particularly for datasets like CIFAR-10, where spatial relationships and patterns are critical.


# Future Enhancements
Experiment with advanced CNN architectures like ResNet and VGG.
Incorporate data augmentation to improve model performance.
Explore transfer learning techniques.

# Contributors
Kabir Mittal





