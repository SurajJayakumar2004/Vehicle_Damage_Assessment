"""
Utility Functions for the Vehicle Fraud Detection Project
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple

AUTOTUNE = tf.data.AUTOTUNE


def create_datasets(
    data_dir: str,
    img_size: int,
    batch_size: int,
    validation_split: float = 0.2,
    seed: int = 42,
    shuffle_test: bool = False
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Create train, validation, and test datasets from directory structure.

    Directory structure expected:
    data_dir/
        train/
            Fraud/
            Non-Fraud/
        test/
            Fraud/
            Non-Fraud/
    """
    img_size_tuple = (img_size, img_size)

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset='training',
        seed=seed,
        image_size=img_size_tuple,
        batch_size=batch_size,
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=validation_split,
        subset='validation',
        seed=seed,
        image_size=img_size_tuple,
        batch_size=batch_size,
        label_mode='binary'
    )

    if os.path.exists(test_dir):
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=img_size_tuple,
            batch_size=batch_size,
            label_mode='binary',
            shuffle=shuffle_test
        )
    else:
        test_ds = None

    def prep(ds):
        return ds.cache().prefetch(AUTOTUNE)

    return prep(train_ds), prep(val_ds), (prep(test_ds) if test_ds else None)


def plot_history(history: tf.keras.callbacks.History, save_path: str):
    """Plot training and validation accuracy and loss curves."""
    metrics = history.history
    epochs = range(1, len(metrics['loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, metrics['accuracy'], label='Train Acc')
    if 'val_accuracy' in metrics:
        ax1.plot(epochs, metrics['val_accuracy'], label='Val Acc')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(epochs, metrics['loss'], label='Train Loss')
    if 'val_loss' in metrics:
        ax2.plot(epochs, metrics['val_loss'], label='Val Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
