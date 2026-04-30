# CNN from Scratch (MNIST)

## Overview

This project implements a simple convolutional neural network pipeline **from scratch** — without using any deep learning frameworks for training.

The system processes handwritten digit images (0–9) from the MNIST dataset and classifies them using manually implemented convolution, pooling, and a softmax classifier.

---

## Architecture

![CNN Architecture](images/cnn_architecture.png)

### Pipeline:
**Input (28×28)** → Convolution (3×3 edge kernel) → ReLU → Pooling (2×2, stride 2) → Convolution → Pooling → Flatten (7×7 → 49) → Dense Layer (49 → 10) → Softmax

**Note:**  
- Feature extraction (convolution + pooling) is **not trainable** (fixed edge-detection kernels).  
- Only the final dense layer is trained using gradient descent.

---

## Training Progress

![Training Curve](images/training_curve.png)

Training was performed over multiple epochs. The accuracy improves significantly as the model learns.

---

## Predictions

![Predictions](images/predictions.png)

Each image shows:
- **T** = True label
- **P** = Predicted label
- Confidence score

---

## Features

- Manual convolution layer (fixed edge detection kernel)
- 2×2 max pooling with stride 2
- Flattening to feature vector
- Softmax classifier with cross-entropy loss
- Custom training loop (no PyTorch, TensorFlow, etc.)

---

## Results

- Achieves **~70% accuracy** on the MNIST test set after a few epochs  
- Demonstrates a full end-to-end learning pipeline with only basic NumPy

---

## How to Run

```bash
pip install -r requirements.txt
python CNN.py
