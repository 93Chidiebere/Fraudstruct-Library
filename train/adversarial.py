"""Offline adversarial augmentation with explicit generated labels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def adversarial_fit(model, X, y, attack_fn, rounds=1, label_column="is_fraud"):
    """Fit ``model`` on original rows plus explicitly labeled generated rows.

    ``attack_fn`` must return either ``(X_generated, y_generated)`` or a
    DataFrame containing feature columns and ``label_column``. The function
    rejects missing/misaligned labels instead of copying labels from unrelated
    source rows. Call this only in an offline training workflow.
    """
    if not callable(attack_fn):
        raise ValueError("attack_fn must be a callable that returns generated features and labels")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    X_base = X.copy()
    y_base = np.asarray(y).reshape(-1)
    if len(X_base) != len(y_base) or len(y_base) == 0:
        raise ValueError("X and y must be non-empty and have the same number of rows")
    if not np.isin(y_base, [0, 1]).all():
        raise ValueError("y must contain binary labels 0 and 1")

    X_augmented = X_base.copy()
    y_augmented = y_base.copy()
    for _ in range(rounds):
        generated = attack_fn(X_base.copy(), y_base.copy())
        if isinstance(generated, tuple) and len(generated) == 2:
            X_generated, y_generated = generated
            if isinstance(X_base, pd.DataFrame):
                if not isinstance(X_generated, pd.DataFrame):
                    X_generated = pd.DataFrame(X_generated, columns=X_base.columns)
                if list(X_generated.columns) != list(X_base.columns):
                    raise ValueError("generated feature columns must match X in name and order")
            else:
                X_generated = np.asarray(X_generated)
                if X_generated.ndim != 2 or X_generated.shape[1] != X_base.shape[1]:
                    raise ValueError("generated feature width must match X")
            y_generated = np.asarray(y_generated).reshape(-1)
        elif isinstance(generated, pd.DataFrame) and label_column in generated.columns:
            y_generated = generated[label_column].to_numpy()
            X_generated = generated.drop(columns=[label_column])
            if not isinstance(X_base, pd.DataFrame):
                raise ValueError("DataFrame attack output requires DataFrame X")
            if list(X_generated.columns) != list(X_base.columns):
                raise ValueError("generated feature columns must match X in name and order")
        else:
            raise ValueError(
                "attack_fn must return (X_generated, y_generated) or a labeled DataFrame"
            )

        if len(X_generated) == 0 or len(X_generated) != len(y_generated):
            raise ValueError("generated features and labels must be non-empty and aligned")
        if not np.isin(y_generated, [0, 1]).all():
            raise ValueError("generated labels must be binary 0/1 values")
        if isinstance(X_base, pd.DataFrame):
            X_augmented = pd.concat([X_augmented, X_generated], ignore_index=True)
        else:
            X_augmented = np.concatenate([X_augmented, X_generated], axis=0)
        y_augmented = np.concatenate([y_augmented, y_generated])
        model.fit(X_augmented, y_augmented)
    return model
