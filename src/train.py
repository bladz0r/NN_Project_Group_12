import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from src.preprocessing import create_folds


def train_baseline(features, labels):
    accuracies = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        create_folds(features, labels), start=1
    ):
        X_train = features[train_idx]
        X_val = features[val_idx]

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Train baseline classifier
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        # Evaluate on validation fold
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average="macro")

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"Fold {fold}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

    print(f"\nMean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Accuracy:  {np.std(accuracies):.4f}")
    print(f"Mean Macro F1: {np.mean(f1_scores):.4f}")