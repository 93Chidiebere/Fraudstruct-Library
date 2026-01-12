from fraudstruct.simulate.splitting import simulate_splitting

def adversarial_fit(
    model,
    X,
    y,
    attack_fn=simulate_splitting,
    rounds=2
):
    X_adv = X.copy()
    y_adv = y.copy()

    for _ in range(rounds):
        adv = attack_fn(X_adv)
        X_adv = X_adv.append(adv, ignore_index=True)
        y_adv = y_adv.append(y_adv.iloc[:len(adv)], ignore_index=True)

        model.fit(X_adv, y_adv)

    return model
