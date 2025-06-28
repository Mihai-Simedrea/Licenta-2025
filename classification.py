import argparse
import datetime
import numpy as np
import tensorflow as tf

from config import (
    DIRECTORY_PATH,
    FILE_PATTERN,
    MAX_WORKERS,
)
from file_processor import BreastCancerDataLoader
from data_processing import (
    RotationHandler,
    PaddingHandler,
    ResizeHandler,
    NormalizeHandler,
    BinaryMaskHandler,
)
from data_pipeline import apply_pipeline
from architectures.classification_architecture import PatchClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


HANDLER_MAP = {
    "rotation": RotationHandler,
    "padding": PaddingHandler,
    "resize": lambda: ResizeHandler(target_height=224, target_width=224),
    "normalize": NormalizeHandler,
    "binarymask": BinaryMaskHandler,
}

def build_pipeline(handler_names):
    chain = None
    for handler_name in reversed(handler_names):
        handler_cls = HANDLER_MAP.get(handler_name.lower())
        if handler_cls is None:
            raise ValueError(f"Unknown handler: {handler_name}")
        handler_instance = handler_cls() if callable(handler_cls) else handler_cls
        if chain:
            handler_instance.set_next(chain)
        chain = handler_instance
    return chain

def generator_from_indices(indices_list, results, num_classes):
    for idx in indices_list:
        img = results[idx].content
        label = results[idx].label
        yield img, tf.one_hot(label, num_classes)

def main(args):
    file_processor = BreastCancerDataLoader(
        DIRECTORY_PATH,
        FILE_PATTERN,
        max_workers=MAX_WORKERS,
    )
    images, patches, num_classes = file_processor.read_mammograms(limit=args.limit)

    handler_names = [h.strip() for h in args.pipeline.split(",") if h.strip()]
    processing_chain = build_pipeline(handler_names)
    results = apply_pipeline(patches, processing_chain)

    indices = list(range(len(results)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.33, random_state=42)

    steps_per_epoch = len(train_idx) // args.batch_size
    val_steps = len(val_idx) // args.batch_size

    train_dataset = (
        tf.data.Dataset.from_generator(
            lambda: generator_from_indices(train_idx, results, num_classes),
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
            ),
        )
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_generator(
            lambda: generator_from_indices(val_idx, results, num_classes),
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.float32),
            ),
        )
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = PatchClassifier(
        input_shape=(224, 224, 3),
        num_patch_classes=2,
        learning_rate=args.learning_rate,
    )
    model.build()
    model = model.compile()

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("model.keras", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
    ]

    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
    )

    y_true = []
    y_pred = []

    for images_batch, labels_batch in val_dataset:
        y_true.append(labels_batch.numpy())
        preds = model.predict(images_batch)
        y_pred.append(preds)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    report = classification_report(
        y_true_classes, y_pred_classes, target_names=["BENIGN", "MALIGNANT"]
    )
    print("Classification Report:\n", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Breast Cancer Classifier")

    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="resize,normalize",
        help="Comma-separated list of pipeline steps (rotation,padding,resize,normalize,binarymask)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Number of Images"
    )

    args = parser.parse_args()
    main(args)
