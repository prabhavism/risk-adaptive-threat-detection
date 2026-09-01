"""
Heavy DL: deeper MLP for flows XGBoost is genuinely uncertain about (conf < THETA_LOW).
Should be measurably slower but more accurate on ambiguous/hard flows.
Person 3's benchmark will measure the latency difference vs Light DL.

Architecture: Input(17) → 128 → BN → Drop(0.3) → 64 → BN → Drop(0.3)
                        →  32 → BN → Drop(0.2) → 16 → BN → 7-softmax

Upgrade path (when Person 1 delivers per-host timestamped flows):
    Replace this MLP with an LSTM that ingests a sequence of the last N flows
    from the same src_ip — much stronger signal for beaconing / exfiltration.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks as keras_callbacks

from ml_dl.config import (
    FEATURE_COLUMNS, CLASSES, HEAVY_DL_PATH,
    HEAVY_EPOCHS, BATCH_SIZE, PATIENCE,
)


# ── Architecture ──────────────────────────────────────────────────────────────

def build_heavy_dl(
    input_dim: int = len(FEATURE_COLUMNS),
    num_classes: int = len(CLASSES),
) -> tf.keras.Model:
    # Exponential decay learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=500,
        decay_rate=0.9,
        staircase=True,
    )

    model = models.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(num_classes, activation="softmax"),
    ], name="heavy_mlp")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_train, y_train, X_val, y_val) -> tf.keras.Model:
    """
    Trains the Heavy MLP. Saves the best checkpoint to HEAVY_DL_PATH.
    Returns the best model (restored by EarlyStopping).
    """
    model = build_heavy_dl()

    cb = [
        keras_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras_callbacks.ModelCheckpoint(
            filepath=str(HEAVY_DL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        # NOTE: ReduceLROnPlateau is intentionally omitted here.
        # The ExponentialDecay schedule already handles LR reduction;
        # combining both causes a TypeError in Keras >= 3.
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=HEAVY_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1,
    )
    return model


# ── Save / load ───────────────────────────────────────────────────────────────

def save(model: tf.keras.Model, path=HEAVY_DL_PATH):
    """Save full model (architecture + weights) in .keras format."""
    model.save(str(path))


def load(path=HEAVY_DL_PATH) -> tf.keras.Model:
    """Load a previously saved Heavy DL model."""
    return tf.keras.models.load_model(str(path))
