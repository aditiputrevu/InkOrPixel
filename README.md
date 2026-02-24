# InkOrPixel 🎨🧠

A deep learning project that classifies artwork as **Traditional** or **Digital** using Convolutional Neural Networks (CNNs).

---

## 📌 Overview

InkOrPixel investigates whether neural networks can distinguish *how* an artwork was created rather than *what* it depicts.

Instead of focusing on semantic content, the model aims to learn subtle visual signals such as:

- Surface texture (e.g., paper grain)
- Stroke irregularity
- Noise distribution
- Gradient smoothness
- Edge sharpness

This frames the task as a **representation learning problem**, where micro-level features are more important than object recognition.

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
