# InkOrPixel 🎨🧠  
### Teaching a Neural Network to See the Difference Between Paper and Pixels

---

## 📌 Overview

**InkOrPixel** is a deep learning project that classifies artwork as **Traditional** or **Digital** using Convolutional Neural Networks.

Unlike typical image classification tasks that focus on *what* is in an image, this project focuses on **how the image was created**.

The model learns subtle visual cues such as:
- paper texture and grain  
- stroke irregularity  
- noise patterns  
- gradient smoothness  
- edge sharpness  

This frames the task as a **representation learning problem**, where low-level features matter more than object semantics.

---

## 🎯 Objective

To build a model that distinguishes artistic medium using texture-level characteristics rather than high-level content.

---

## 🧠 Approach

### Data Processing
- Custom dataset of digital vs traditional artwork  
- Train / validation / test split  
- Data augmentation (rotation, flipping, color jitter)

### Models

#### 🔹 Baseline CNN  
A simple CNN to establish baseline performance.

#### 🔹 Transfer Learning (ResNet18)  
- Adapted for binary classification  
- Fine-tuned final layers  
- Used when pretrained weights are available  

---

## ⚙️ Tech Stack

- Python  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  

---

## 📊 Results

| Test Case | Prediction | Confidence |
|----------|-----------|-----------|
| Image 1 | Digital | **87.44%** |
| Image 2 | Traditional | **58.03%** |

### Observations
- High confidence on clear cases  
- Moderate confidence on ambiguous images  
- Performance limited by dataset size and training constraints  

---

## Project Structure

```
InkOrPixel/
│
├── main.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── model.py
│   └── config.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── models/
│   └── plots/
│
├── notebooks/
│
└── assets/
```
---

## 🎯 Objective

To design and train deep learning models that classify artistic medium based on texture-level characteristics.

---

## 🧠 Models

### Baseline CNN  
A custom convolutional neural network implemented to establish a performance baseline.

### Transfer Learning (ResNet18)  
A pretrained ResNet model adapted for binary classification to improve feature extraction and generalization.

---

## ⚙️ Tech Stack

- Python 3.x  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  

---

## 📂 Current Status

Project setup and model scaffolding in progress.  
Training and evaluation pipeline under development.

---
