# Nearest Neighbor Image Classifier — CIFAR-10

A from-scratch implementation of the **Nearest Neighbor classifier**, the simplest possible image classification algorithm, applied to the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Built as a baseline exercise to understand the fundamentals of image classification before moving to more advanced approaches (KNN, linear classifiers, CNNs).

## Overview

Nearest Neighbor does not "learn" in the traditional sense — it simply **memorizes** every training image, and for each test image, finds the single most similar training image (by raw pixel distance) and copies its label.

```
train(images, labels)   -> memorize all data and labels           [O(1)]
predict(model, images)  -> find nearest training image, copy label [O(N) per prediction]
```

This project is intentionally simple. The goal isn't strong accuracy — it's to establish a baseline and expose two problems that motivate every algorithm that comes after it:

1. **Inverted speed tradeoff** — training is instant, but prediction requires scanning the *entire* training set every time. Real-world systems need the opposite: slow offline training, fast real-time inference.
2. **Raw pixel distance is a poor similarity measure** — two images of the same object can have huge pixel-distance differences due to lighting, shifting, or background, while unrelated images can appear "closer." This motivates learning actual visual features instead of comparing pixels directly.

## Dataset

**CIFAR-10**: 60,000 32×32 color images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) — 50,000 training images, 10,000 test images.

The dataset is downloaded automatically on first run from the [official source](https://www.cs.toronto.edu/~kriz/cifar.html) (~170MB) and cached locally in `cifar10_data/`.

## Project Structure

```
.
├── nearest_neighbor_classifier.py     # standalone script version
├── nearest_neighbor_classifier.ipynb  # Jupyter notebook version (recommended)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/meghaInfosec/KNN-based-CIFAR10-Image-Classification.git
cd KNN-based-CIFAR10-Image-Classification

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

**Option A — Jupyter Notebook (recommended, includes visualizations):**
```bash
jupyter notebook nearest_neighbor_classifier.ipynb
```
Run all cells in order. The first data-loading cell downloads CIFAR-10 automatically.

**Option B — Standalone script:**
```bash
python3 nearest_neighbor_classifier.py
```

## How It Works

| Step | Function | Complexity | What it does |
|------|----------|------------|---------------|
| Train | `train(images, labels)` | O(1) | Stores a reference to the training images and labels — no computation |
| Predict | `predict(test_images)` | O(N) per test image | Computes pixel-wise distance (L1 or L2) between the test image and every training image; returns the label of the closest match |

Distance metrics supported:
- **L1 (Manhattan distance)**: sum of absolute pixel differences
- **L2 (Euclidean distance)**: square root of sum of squared pixel differences

## Results

Using a subset of 5,000 training images and 200 test images with L1 distance:

| Metric | Value |
|--------|-------|
| Accuracy | ~25–35% |
| Random guess baseline | 10% (10 classes) |

Accuracy is intentionally modest — this confirms the algorithm's core limitation: comparing raw pixels does not capture meaningful visual similarity. This result is the expected motivation for moving to KNN, linear classifiers, and eventually CNNs, which learn actual features instead of comparing pixels directly.

## Decision Surface Visualization

Since CIFAR-10 images live in 3072 dimensions (32×32×3 pixels), the decision boundary can't be plotted directly. The notebook includes a section that:
1. Reduces a subset of images to 2D using PCA
2. Fits the Nearest Neighbor classifier directly in that 2D space
3. Predicts labels across a fine grid to visualize the resulting decision surface
4. Overlays the actual training points, colored by class

The result is a Voronoi-style diagram showing the jagged, cell-like boundaries Nearest Neighbor creates — useful for building intuition about why NN is sensitive to outliers, even though it's a 2D approximation of the much higher-dimensional real decision surface.

## Notes

- The full 50,000-image training set is not used by default because pure-numpy nearest-neighbor search does not scale well — prediction time grows linearly with training set size, which is one of the core limitations this project is meant to demonstrate.
- No GPU or deep learning framework is required — this is pure NumPy.

## References

- CIFAR-10 dataset: [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
- Stanford CS231n: Convolutional Neural Networks for Visual Recognition
