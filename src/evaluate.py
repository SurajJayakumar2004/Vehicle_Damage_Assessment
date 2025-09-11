"""
Model Evaluation Script

Evaluates a pre-trained model on the test dataset and outputs
classification metrics and a confusion matrix plot.
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained fraud detection model")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved .h5 or SavedModel directory')
    parser.add_argument('--data_dir', type=str, required=True, help='Root data directory containing test/ subfolder')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='reports')
    return parser.parse_args()


def get_true_labels(dataset: tf.data.Dataset):
    labels = []
    for _, y in dataset:  # batches
        labels.extend(y.numpy().astype(int).tolist())
    return np.array(labels)


def evaluate():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'figures'), exist_ok=True)

    print(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # Only need test dataset; use util to construct all and pick test
    _, _, test_ds = utils.create_datasets(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        validation_split=0.2,
        shuffle_test=False
    )

    if test_ds is None:
        raise ValueError("Test dataset could not be created. Ensure data_dir/test exists.")

    # Collect predictions
    y_true = get_true_labels(test_ds)
    y_prob = model.predict(test_ds).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    target_names = ['Non-Fraud', 'Fraud']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print(report)

    # Save classification report
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    fig_path = os.path.join(args.output_dir, 'figures', 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Confusion matrix saved to {fig_path}")


if __name__ == '__main__':
    evaluate()
