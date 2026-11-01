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

        deltas = np.diff(recording, axis=0)
        mean_abs_deltas = np.mean(np.abs(deltas), axis=0)

        features.append(
            np.concatenate([means, stds, mean_abs_deltas])
        )

    return np.array(features)


def extract_segmented_features(inputs, n_segments=3):
    features = []

    for recording in inputs:
        segments = np.array_split(recording, n_segments, axis=0)
        recording_features = []

        for segment in segments:
            means = np.mean(segment, axis=0)
            stds = np.std(segment, axis=0)

            recording_features.extend(means)
            recording_features.extend(stds)

        features.append(recording_features)

    return np.array(features)

