"""
Heavy DL: deeper verification network for flows XGBoost is uncertain
about. Should be measurably slower than Light DL but more accurate on
ambiguous flows — that trade-off is the whole point of the benchmark
Person 3 runs at the end.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from ml_dl.config import FEATURE_COLUMNS, CLASSES, HEAVY_DL_PATH


def build_heavy_dl(input_dim: int = len(FEATURE_COLUMNS),
                    num_classes: int = len(CLASSES)) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save(model: tf.keras.Model, path=HEAVY_DL_PATH):
    model.save_weights(path)


def load(path=HEAVY_DL_PATH):
    model = build_heavy_dl()
    model.load_weights(path)
    return model
