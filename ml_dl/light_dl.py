"""
Light DL: small, fast verification network for flows XGBoost is already
confident about. Architecture is finalized now against synthetic data
shape; weights get retrained once real data arrives.
"""
import tensorflow as tf
from tensorflow.keras import layers, models

from ml_dl.config import FEATURE_COLUMNS, CLASSES, LIGHT_DL_PATH


def build_light_dl(input_dim: int = len(FEATURE_COLUMNS),
                    num_classes: int = len(CLASSES)) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save(model: tf.keras.Model, path=LIGHT_DL_PATH):
    model.save_weights(path)


def load(path=LIGHT_DL_PATH):
    model = build_light_dl()
    model.load_weights(path)
    return model
