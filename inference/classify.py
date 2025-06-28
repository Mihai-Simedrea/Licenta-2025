import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from data_processing import (
    ResizeHandler,
    NormalizeHandler,
)

from data_pipeline import apply_pipeline

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0

    return img

def main(args):
    model = tf.keras.models.load_model(args.model_path)
    img = preprocess_image(args.image_path)
    img_batch = np.expand_dims(img, axis=0)

    preds = model.predict(img_batch)
    pred_class = np.argmax(preds, axis=1)[0]
    class_names = ["BENIGN", "MALIGNANT"]

    print(f"Predicted class: {class_names[pred_class]} (probabilities: {preds[0]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a mammogram patch")

    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image to classify"
    )
    parser.add_argument(
        "--model_path", type=str, default="model.keras", help="Path to the trained model"
    )

    args = parser.parse_args()
    main(args)
