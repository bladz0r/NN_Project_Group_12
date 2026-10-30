import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.preprocessing import create_folds


def _get_trained_mlp(X_train, y_train, alpha=0.5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="tanh",
        alpha=alpha,
        solver="adam",
        max_iter=1000,
        random_state=42,  # seed 42 like always :)
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


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


def train_mlp(features, labels, alpha=0.5):
    accuracies = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        create_folds(features, labels), start=1
    ):
        X_train = features[train_idx]
        X_val = features[val_idx]

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        model, scaler = _get_trained_mlp(X_train, y_train, alpha=alpha)
        # Standardize features
        X_val_scaled = scaler.transform(X_val)

        # Evaluate on validation fold
        predictions = model.predict(X_val_scaled)

        # Calculate scores and save to variables
        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average="macro")

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(f"MLP Fold {fold}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}")

    print(f"\nMLP Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"MLP Std Accuracy:  {np.std(accuracies):.4f}")
    print(f"MLP Mean Macro F1: {np.mean(f1_scores):.4f}")


def evaluate_test_mlp(
    train_features, train_labels, test_features, test_labels, alpha=0.5
):
    model, scaler = _get_trained_mlp(train_features, train_labels, alpha=alpha)
    # we scale the features using the training set
    X_test_scaled = scaler.transform(test_features)
    predictions = model.predict(X_test_scaled)

    # here we calculate the final metrics
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average="macro")
    misclassification_rate = (1 - accuracy) * 100

    print("\n--- The final test set evaluation (MLP) ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {f1:.4f}")
    print(f"Misclassification Rate: {misclassification_rate:.2f}%")


def plot_mlp_confusion_matrix(
    train_features, train_labels, test_features, test_labels, alpha=0.5
):
    model, scaler = _get_trained_mlp(train_features, train_labels, alpha=alpha)
    X_test_scaled = scaler.transform(test_features)

    predictions = model.predict(X_test_scaled)
    cm = confusion_matrix(test_labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    plt.title("MLP Classifier - Test Set Confusion Matrix")
    plt.xlabel("Predicted Speaker Class")
    plt.ylabel("True Speaker Class")
    plt.tight_layout()
    plt.show()
