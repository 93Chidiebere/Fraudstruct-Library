import numpy as np


def evaluate_blindspots(model, X, y, adversarial_flags):
    preds = model.predict(X)

    blindspot_mask = (preds == 0) & (adversarial_flags == 1)
    blindspot_rate = np.mean(blindspot_mask)

    return {
        "blindspot_rate": blindspot_rate,
        "blindspot_count": blindspot_mask.sum()
    }
