from sklearn.metrics import roc_auc_score

def robustness(model, X_clean, y_clean, X_adv, y_adv):
    return {
        "auc_clean": roc_auc_score(y_clean, model.predict_proba(X_clean)[:, 1]),
        "auc_adv": roc_auc_score(y_adv, model.predict_proba(X_adv)[:, 1])
    }
