from sklearn.ensemble import IsolationForest

def train_model(X):
    """
    Train Isolation Forest on normal operating data only.
    """
    model = IsolationForest(
        max_samples=256,
        n_estimators=100,
        contamination=0.05,
        random_state=42,
        verbose=1
    )
    print("Training Isolation Forest...")
    model.fit(X)
    print("Isolation Forest trained successfully")
    return model

def anomaly_score(model, X):
    # Higher = more abnormal
    scores = -model.decision_function(X)
    return scores
