"""
Main Training Script for Vehicle Damage Fraud Detection

This script orchestrates the entire training pipeline:
1.  Parses command-line arguments for hyperparameters and paths.
2.  Loads and preprocesses the training and validation data using functions from utils.py.
3.  Calculates class weights to handle dataset imbalance.
4.  Builds the transfer learning model (e.g., EfficientNetV2-B0).
5.  Compiles the model with the appropriate optimizer, loss function, and metrics.
6.  Sets up callbacks for early stopping and saving the best model.
7.  Trains the model.
8.  Saves the final, best-performing model to the specified output path.
"""

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

import utils  # local utils module

# --------------------------------------------------
# Model Builder
# --------------------------------------------------

def build_model(img_size: int, learning_rate: float = 1e-4, dropout: float = 0.3) -> keras.Model:
    """Builds a transfer learning model using EfficientNetV2B0.

    Args:
        img_size: Target width/height (square) for input images.
        learning_rate: Learning rate for optimizer.
        dropout: Dropout rate after pooling.

    Returns:
        A compiled Keras model ready for training.
    """
    input_shape = (img_size, img_size, 3)
    base_model = keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        input_shape=input_shape,
        weights="imagenet"
    )
    base_model.trainable = False  # freeze feature extractor initially

    inputs = layers.Input(shape=input_shape)
    x = keras.applications.efficientnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs, name="efficientnetv2b0_fraud")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

# --------------------------------------------------
# Utility: Class Weights
# --------------------------------------------------

def compute_class_weights_from_dataset(dataset: tf.data.Dataset) -> dict:
    """Compute class weights for a binary dataset.

    Expects dataset yielding (images, labels) where labels are 0/1.
    """
    labels = []
    for _, y in dataset.unbatch():
        labels.append(int(y.numpy()))
    labels = np.array(labels)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    return class_weight

# --------------------------------------------------
# Argument Parser
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-B0 for Vehicle Fraud Detection")
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory containing train/ and test/ folders')
    parser.add_argument('--model_output_path', type=str, default='models/fraud_efficientnetv2b0.h5', help='Path to save best model')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--unfreeze_after', type=int, default=0, help='Epoch after which to unfreeze base model (0 disables)')
    return parser.parse_args()

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.model_output_path), exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    train_ds, val_ds, test_ds = utils.create_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=args.val_split
    )

    class_weight = compute_class_weights_from_dataset(train_ds)
    print("Computed class weights:", class_weight)

    model = build_model(img_size=args.img_size, learning_rate=args.learning_rate, dropout=args.dropout)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=args.model_output_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-7
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )

    utils.plot_history(history, 'reports/figures/training_history.png')

    # Optional fine-tuning
    if args.unfreeze_after > 0:
        print(f"Starting fine-tuning after epoch {args.unfreeze_after}...")
        base_model = model.layers[2]  # input, preprocess, base_model index assumption
        base_model.trainable = True
        # Recompile with lower LR
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate * 0.1),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=[
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=max(5, args.epochs // 3),
            class_weight=class_weight,
            callbacks=callbacks
        )
        utils.plot_history(ft_history, 'reports/figures/fine_tune_history.png')

    model.save(args.model_output_path)
    print(f"Model saved to {args.model_output_path}")

    # Basic evaluation on held-out test set if available
    if test_ds is not None:
        print("Evaluating on test set...")
        results = model.evaluate(test_ds)
        print("Test results:", dict(zip(model.metrics_names, results)))

if __name__ == '__main__':
    main()
