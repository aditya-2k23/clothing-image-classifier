import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import time
import os

# Class names for Fashion-MNIST
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def load_data():
    """Loads Fashion-MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, X_test, y_train, y_test


def show_samples(X, y, num=20):
    """Displays sample images for EDA."""
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(4, 5, i + 1)
        plt.imshow(X[i], cmap="gray")
        plt.title(CLASS_NAMES[y[i]])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def pixel_intensity_distribution(X):
    """Plots histogram of pixel intensities."""
    plt.figure(figsize=(8, 4))
    plt.hist(X.flatten(), bins=50, color="gray")
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()


def plot_class_mean_images(X, y):
    """Plots the average image of each class."""
    plt.figure(figsize=(10, 4))
    for cls in range(10):
        mean_img = X[y == cls].mean(axis=0)
        plt.subplot(2, 5, cls + 1)
        plt.imshow(mean_img, cmap="gray")
        plt.title(CLASS_NAMES[cls])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def summary_statistics(X):
    """Prints basic summary statistics."""
    print("=== Summary Statistics ===")
    print(f"Mean pixel value: {np.mean(X):.2f}")
    print(f"Std deviation: {np.std(X):.2f}")
    print("Min pixel value:", np.min(X))
    print("Max pixel value:", np.max(X))


def preprocess_for_sklearn(X_train, X_test):
    """
    Scales images to [0,1] and flattens into 784 features.
    Suitable for Logistic Regression & Random Forest.
    """
    X_train_scaled = X_train.astype("float32") / 255.0
    X_test_scaled = X_test.astype("float32") / 255.0

    X_train_flat = X_train_scaled.reshape(len(X_train_scaled), -1) # 60000 x (28*28)
    X_test_flat = X_test_scaled.reshape(len(X_test_scaled), -1) # 10000 x (28*28)

    return X_train_flat, X_test_flat


def preprocess_for_cnn(X_train, X_test):
    """
    Scales images to [0,1] and reshapes to (28, 28, 1).
    Suitable for CNN.
    """
    X_train_scaled = X_train.astype("float32") / 255.0
    X_test_scaled = X_test.astype("float32") / 255.0

    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], 28, 28, 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], 28, 28, 1)

    return X_train_cnn, X_test_cnn


def train_with_timing(model, X_train, y_train):
    """
    Trains any sklearn/Keras model while measuring training time.
    
    Returns:
        trained_model
        training_time (in seconds)
    """
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    training_time = end - start
    return model, training_time


def get_model_size(model_path):
    """
    Returns file size of saved model in kilobytes (KB).
    
    Parameters:
        model_path (str): Path to the saved model file (.joblib, .pkl, .keras)

    Returns:
        size_kb (float): File size in KB
    """
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    return size_kb

def train_cnn_with_timing(model, X_train, y_train, epochs=10, batch_size=128):
    """
    Trains a Keras CNN model while measuring ONLY the training time.

    Returns:
        history   : Keras History object
        train_time: float (seconds)
    """
    import time

    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    end = time.time()

    return history, end - start
