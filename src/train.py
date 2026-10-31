import numpy as np

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from src.mlp import create_mlp



def fit_mlp(X_train, y_train, **mlp_params):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = create_mlp()
    model.set_params(**mlp_params)

    model.fit(X_train_scaled, y_train)
    return model, scaler



def train_mlp_with_history(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=300,
    **mlp_params,
):
    # Standardize features using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Define MLP architecture
    model = create_mlp()
    model.set_params(**mlp_params)

    # Store losses for each epoch
    history = {
        "train_loss": [],
        "val_loss": [],
    }

    classes = np.unique(y_train)

    # Train epoch by epoch
    for epoch in range(epochs):

        # First epoch requires all training classes
        if epoch == 0:
            model.partial_fit(X_train_scaled, y_train, classes=classes)
        else:
            model.partial_fit(X_train_scaled, y_train)

        # Predict class probabilities
        train_probs = model.predict_proba(X_train_scaled)
        val_probs = model.predict_proba(X_val_scaled)

        # Calculate cross-entropy loss
        train_loss = log_loss(
            y_train, train_probs, labels=model.classes_
        )
        val_loss = log_loss(
            y_val, val_probs, labels=model.classes_
        )

        # Store losses
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

    return model, scaler, history


def partial_fit_mlp(X_train, y_train, epochs=300, **mlp_params):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = create_mlp()
    model.set_params(**mlp_params)

    classes = np.unique(y_train)

    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(X_train_scaled, y_train, classes=classes)
        else:
            model.partial_fit(X_train_scaled, y_train)

    return model, scaler