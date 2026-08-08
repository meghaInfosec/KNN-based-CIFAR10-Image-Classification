"""
Nearest Neighbor Classifier on CIFAR-10
=========================================
This is the classic baseline classifier from Stanford CS231n.
It does NOT use any deep learning -- it just memorizes training
images and, for every test image, finds the single most similar
training image (by raw pixel distance) and copies its label.

Motive: to establish a rock-bottom baseline that shows WHY we need
smarter models (KNN -> Linear classifiers -> CNNs). See explanation
at the bottom of this file / in the chat response.
"""

import numpy as np
import pickle
import os
import urllib.request
import tarfile


# ----------------------------------------------------------------------
# 1. Load CIFAR-10 (downloads the official batches if not already present)
# ----------------------------------------------------------------------
def download_cifar10(data_dir="./cifar10_data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")

    if os.path.exists(extracted_dir):
        return extracted_dir

    os.makedirs(data_dir, exist_ok=True)
    print("Downloading CIFAR-10 (~170MB)...")
    urllib.request.urlretrieve(url, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)

    return extracted_dir


def load_cifar_batch(filename):
    with open(filename, "rb") as f:
        batch = pickle.load(f, encoding="bytes")
        images = batch[b"data"]                     # shape: (10000, 3072)
        labels = batch[b"labels"]
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC
        return images, np.array(labels)


def load_cifar10(data_dir="./cifar10_data"):
    folder = download_cifar10(data_dir)

    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_cifar_batch(os.path.join(folder, f"data_batch_{i}"))
        xs.append(x)
        ys.append(y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)

    X_test, y_test = load_cifar_batch(os.path.join(folder, "test_batch"))

    return X_train, y_train, X_test, y_test


# ----------------------------------------------------------------------
# 2. The Nearest Neighbor Classifier itself
# ----------------------------------------------------------------------
class NearestNeighbor:
    def __init__(self):
        self.Xtr = None
        self.ytr = None

    def train(self, images, labels):
        """
        'Machine learning!' -- but really just memorization.
        O(1): no computation, just store a reference to the data.
        """
        self.Xtr = images
        self.ytr = labels

    def predict(self, test_images, distance="L1"):
        """
        For every test image, scan ALL training images and return
        the label of whichever one is closest (most similar).
        O(N) per prediction -- this is the expensive part.
        """
        num_test = test_images.shape[0]
        y_pred = np.zeros(num_test, dtype=self.ytr.dtype)

        # Flatten images so each one is a single long vector of pixels
        Xtr_flat = self.Xtr.reshape(self.Xtr.shape[0], -1).astype(np.float64)

        for i in range(num_test):
            test_vec = test_images[i].reshape(-1).astype(np.float64)

            if distance == "L1":
                # Sum of absolute pixel differences
                distances = np.sum(np.abs(Xtr_flat - test_vec), axis=1)
            elif distance == "L2":
                # Euclidean distance
                distances = np.sqrt(np.sum((Xtr_flat - test_vec) ** 2, axis=1))
            else:
                raise ValueError("distance must be 'L1' or 'L2'")

            nearest_index = np.argmin(distances)   # index of the closest match
            y_pred[i] = self.ytr[nearest_index]    # copy its label

            if (i + 1) % 50 == 0:
                print(f"  predicted {i + 1}/{num_test} test images...")

        return y_pred


# ----------------------------------------------------------------------
# 3. Run it end-to-end
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading CIFAR-10...")
    X_train, y_train, X_test, y_test = load_cifar10()

    # Using a small subset here (full 50k vs 10k would take a long time
    # in pure numpy -- this is exactly the weakness we're demonstrating!)
    N_TRAIN = 5000
    N_TEST = 200

    X_train, y_train = X_train[:N_TRAIN], y_train[:N_TRAIN]
    X_test, y_test = X_test[:N_TEST], y_test[:N_TEST]

    nn = NearestNeighbor()

    print("Training (memorizing data)...")
    nn.train(X_train, y_train)   # instant

    print("Predicting (comparing every test image against all training images)...")
    y_pred = nn.predict(X_test, distance="L1")   # slow

    accuracy = np.mean(y_pred == y_test)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
