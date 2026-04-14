# InkOrPixel 🎨🧠
### Teaching a Neural Network to See the Difference Between Paper and Pixels

---

## 📌 Overview

**InkOrPixel** classifies artwork as **Traditional** or **Digital** using Convolutional Neural Networks.

Unlike typical image classification that focuses on *what* is in an image, this project focuses on **how the image was created** — framing it as a representation learning problem where low-level features matter more than object semantics.

The model learns subtle visual cues such as:
- Paper texture and grain
- Stroke irregularity
- Noise patterns
- Gradient smoothness
- Edge sharpness

---

## 🎯 Objective

To build a model that distinguishes artistic medium using texture-level characteristics rather than high-level content.

---

## 🧠 Approach

### Data Processing
- Custom dataset of digital vs. traditional artwork
- Train / validation / test split
- Data augmentation (rotation, flipping, color jitter)

### Models

#### 🔹 Baseline CNN
A simple custom convolutional neural network trained from scratch to establish a performance baseline.

#### 🔹 Transfer Learning (ResNet18)
- Pretrained on ImageNet, adapted for binary classification
- Fine-tuned final layers
- Used when pretrained weights are available

---

## ⚙️ Tech Stack

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- Matplotlib

---

## 📊 Results

| Test Case | Prediction  | Confidence  |
|-----------|-------------|-------------|
| Image 1   | Digital     | **87.44%**  |
| Image 2   | Traditional | **58.03%**  |

### Observations
- High confidence on clear examples
- Moderate confidence on ambiguous images (e.g. clean line art)
- Performance limited by dataset size and training constraints

---

## 🧪 Example Usage

```bash
python main.py predict --image path/to/image.jpg
```

**Output:**
```
Prediction: traditional
Confidence: 57.91%
```

---

## 📂 Project Structure

```
InkOrPixel/
│
├── main.py
├── prepare_data.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── dataset.py
│   ├── model.py
│   ├── config.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│   ├── models/
│   ├── plots/
│   └── predictions/
│
├── notebooks/
└── assets/
```

---

## 🚀 How to Run

### 1. Prepare dataset
```bash
python prepare_data.py
```

### 2. Train model
```bash
python main.py train
```

### 3. Evaluate model
```bash
python main.py evaluate
```

### 4. Predict on a new image
```bash
python main.py predict --image path/to/image.jpg
```

---

## ⚠️ Challenges

- Small dataset leading to overfitting
- SSL issue preventing consistent pretrained weight usage
- Difficulty distinguishing clean line art from digital artwork

---

## 🔮 Future Improvements

- Larger and more diverse dataset
- Full fine-tuning of deeper ResNet layers
- Improved preprocessing for texture detection

---

## 💡 Key Takeaway

InkOrPixel demonstrates that neural networks can learn *how an image was created*, not just *what it contains* — even with limited data.

---

## 👩‍💻 Author

**Aditi Putrevu**
Northeastern University — Master's in Computer Science