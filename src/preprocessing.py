import numpy as np
from sklearn.model_selection import StratifiedKFold


# ----------Stratified Data Preprocessing with Cross-Validation ----------
def create_folds(train_inputs, train_labels):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return skf.split(train_inputs, train_labels)


def extract_features(inputs):
    features = []

    for recording in inputs:
        means = np.mean(recording, axis=0)
        stds = np.std(recording, axis=0)

        features.append(np.concatenate([means, stds]))

    return np.array(features)