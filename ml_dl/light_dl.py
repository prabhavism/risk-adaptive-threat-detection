"""
Light DL: small, fast MLP verification network.
Runs for flows where XGBoost confidence is moderate (THETA_LOW <= conf < THETA_HIGH).
Designed to be measurably faster than Heavy DL — that trade-off is benchmarked by Person 3.

Architecture: Input(17) → 64 → BN → Drop(0.3) → 32 → BN → Drop(0.2) → 7-softmax
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks as keras_callbacks

from ml_dl.config import (
    FEATURE_COLUMNS, CLASSES, LIGHT_DL_PATH,
    LIGHT_EPOCHS, BATCH_SIZE, PATIENCE,
)


# ── Architecture ──────────────────────────────────────────────────────────────

def build_light_dl(
    input_dim: int = len(FEATURE_COLUMNS),
    num_classes: int = len(CLASSES),
) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ], name="light_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val) -> tf.keras.Model:
    """
    Trains the Light MLP. Saves the best checkpoint to LIGHT_DL_PATH.
    Returns the best model (restored by EarlyStopping).
    """
    model = build_light_dl()

    cb = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=str(LIGHT_DL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras_callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=LIGHT_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1,
    )
    return model


# ── Save / load ───────────────────────────────────────────────────────────────

def save(model: tf.keras.Model, path=LIGHT_DL_PATH):
    """Save full model (architecture + weights) in .keras format."""
    model.save(str(path))


def load(path=LIGHT_DL_PATH) -> tf.keras.Model:
    """Load a previously saved Light DL model."""
    return tf.keras.models.load_model(str(path))
