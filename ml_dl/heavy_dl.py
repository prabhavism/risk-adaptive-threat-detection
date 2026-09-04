"""
Heavy DL: deeper *temporal* verification network for flows XGBoost is
uncertain about. This is the model that actually earns its keep on
beaconing / DGA / exfil, where the signal lives in the sequence of
recent flows from a host, not any single flow in isolation.

Input: a sequence of the last SEQ_LEN feature vectors from the same
src_ip (see ml_dl.data_utils.build_sequences), scaled the same way as
Light DL. Should be measurably slower than Light DL but more accurate
on ambiguous flows -- that trade-off is the whole point of the
risk-adaptive routing benchmark (scripts/benchmark.py).
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from ml_dl.config import FEATURE_COLUMNS, CLASSES, HEAVY_DL_PATH, SEQ_LEN


def build_heavy_dl(seq_len: int = SEQ_LEN,
                    input_dim: int = len(FEATURE_COLUMNS),
                    num_classes: int = len(CLASSES),
                    use_attention: bool = False) -> tf.keras.Model:
    """
    Default: two stacked GRUs (first keeps the full sequence for the
    second to look at, second collapses it to a single summary vector).

    use_attention=True swaps the second GRU's collapse for a learned
    attention-weighted pooling over the first GRU's outputs instead of
    just taking the last timestep -- lets the model weight whichever
    flow in the window was most informative rather than always trusting
    the most recent one. Off by default: section 9 explicitly says not
    to assume the more complex option is better without checking
    validation performance/inference cost first -- compare both via
    ml_dl.ablation-style experimentation before switching the default.
    """
    inputs = layers.Input(shape=(seq_len, input_dim))
    x = layers.GRU(64, return_sequences=True, dropout=0.2)(inputs)

    if use_attention:
        scores = layers.Dense(1, activation="tanh")(x)
        scores = layers.Flatten()(scores)
        weights = layers.Activation("softmax")(scores)
        weights = layers.RepeatVector(64)(weights)
        weights = layers.Permute([2, 1])(weights)
        weighted = layers.Multiply()([x, weights])
        pooled = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted)
    else:
        pooled = layers.GRU(32, dropout=0.2)(x)

    x = layers.Dense(32, activation="relu")(pooled)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_heavy_dl(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    epochs: int = 50, batch_size: int = 128,
                    use_attention: bool = False, verbose: int = 1):
    """
    X_train/X_val: shape (n, SEQ_LEN, n_features), already scaled and
    built with ml_dl.data_utils.build_sequences. Returns
    (model, history, val_metrics).
    """
    model = build_heavy_dl(
        seq_len=X_train.shape[1], input_dim=X_train.shape[2],
        use_attention=use_attention,
    )

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


def save(model: tf.keras.Model, path=HEAVY_DL_PATH):
    model.save(path)


def load(path=HEAVY_DL_PATH):
    return tf.keras.models.load_model(path)
