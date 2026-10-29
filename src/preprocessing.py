import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

# ---------- Stratified Data Preprocessing without Cross-Validation ----------
def preprocess_data(train_inputs, train_labels):
    X_train, X_val, y_train, y_val = train_test_split(
        train_inputs,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels
    )

    return X_train, X_val, y_train, y_val

# ----------Stratified Data Preprocessing with Cross-Validation ----------
def create_folds(train_inputs, train_labels):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return skf.split(train_inputs, train_labels)