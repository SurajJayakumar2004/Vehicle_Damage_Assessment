"""
Single Image Prediction Script

Loads a trained model and predicts whether a single image
represents a fraudulent claim (Fraud) or not (Non-Fraud).
"""
import argparse
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = ['Non-Fraud', 'Fraud']  # 0 -> Non-Fraud, 1 -> Fraud


def preprocess_image(image_path: str):
    """Load and preprocess image for prediction."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr


def predict(model: tf.keras.Model, image_tensor: np.ndarray):
    """Generate prediction and confidence."""
    prob = float(model.predict(image_tensor)[0][0])
    label_idx = int(prob >= 0.5)
    confidence = prob if label_idx == 1 else 1 - prob
    return {"class": CLASS_NAMES[label_idx], "confidence": round(confidence, 4), "raw_probability": prob}


def parse_args():
    parser = argparse.ArgumentParser(description="Single image fraud prediction")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = tf.keras.models.load_model(args.model_path)
    image_tensor = preprocess_image(args.image_path)
    result = predict(model, image_tensor)
    print(result)


if __name__ == '__main__':
    main()
