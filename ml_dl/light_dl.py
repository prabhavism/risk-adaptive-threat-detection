"""
Light DL: small, fast verification network run on flows XGBoost is
already fairly confident about (see ml_dl/routing.py). Its job is
cheap high-throughput screening, not squeezing out the last bit of
accuracy -- that's what Heavy DL is for on the harder flows.

Input is a single scaled feature vector (see ml_dl.data_utils / the
StandardScaler saved by train_xgboost.py), not a sequence.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from ml_dl.config import FEATURE_COLUMNS, CLASSES, LIGHT_DL_PATH


def build_light_dl(input_dim: int = len(FEATURE_COLUMNS),
                    num_classes: int = len(CLASSES)) -> tf.keras.Model:
    # Deliberately small (section 8): Light DL exists for cheap, fast
    # screening of high-confidence flows, not to compete with Heavy DL
    # on accuracy. Dropout is enough regularization here; a model this
    # size doesn't need BatchNorm to train stably.
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ], name="light_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_light_dl(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 50, batch_size: int = 128,
                    verbose: int = 1):
    """
    X_train/X_val must already be scaled (StandardScaler from
    train_xgboost.py) -- unlike XGBoost, this MLP is scale-sensitive.
    Returns (model, history, val_metrics).
    """
    model = build_light_dl(input_dim=X_train.shape[1])

    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=6, restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=verbose,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    val_metrics = {"val_loss": float(val_loss), "val_accuracy": float(val_acc)}
    return model, history, val_metrics


def save(model: tf.keras.Model, path=LIGHT_DL_PATH):
    model.save(path)


def load(path=LIGHT_DL_PATH):
    return tf.keras.models.load_model(path)
