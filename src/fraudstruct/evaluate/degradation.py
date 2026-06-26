from sklearn.metrics import roc_auc_score


def evaluate_model_degradation(
    model,
    X_clean,
    y_clean,
    X_adv,
    y_adv
):
    auc_clean = roc_auc_score(y_clean, model.predict_proba(X_clean)[:, 1])
    auc_adv = roc_auc_score(y_adv, model.predict_proba(X_adv)[:, 1])

    return {
        "auc_clean": auc_clean,
        "auc_adversarial": auc_adv,
        "performance_drop": auc_clean - auc_adv
    }
