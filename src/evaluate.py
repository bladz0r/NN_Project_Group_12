import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler

from src.preprocessing import create_folds
from src.train import fit_mlp, train_mlp_with_history, partial_fit_mlp



def cross_validate_baseline(features, labels):
    accuracies = []
    f1_scores = []

    print("\n--- Logistic Regression ---")

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

    results = {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores),
    }
    print_cross_validation_metrics(results)


def cross_validate_mlp(features, labels, **mlp_params):

    accuracies = []
    f1_scores = []
    histories = []

    print("\n--- MLP Classifier ---")

    for fold, (train_idx, val_idx) in enumerate(
        create_folds(features, labels), start=1
    ):
        X_train = features[train_idx]
        X_val = features[val_idx]

        y_train = labels[train_idx]
        y_val = labels[val_idx]

        # Train model and collect losses
        model, scaler, history = train_mlp_with_history(
            X_train,
            y_train,
            X_val,
            y_val,
            **mlp_params,
        )

        histories.append(history)

        # Evaluate on validation fold
        X_val_scaled = scaler.transform(X_val)
        predictions = model.predict(X_val_scaled)

        accuracy = accuracy_score(y_val, predictions)
        f1 = f1_score(y_val, predictions, average="macro")

        accuracies.append(accuracy)
        f1_scores.append(f1)

        print(
            f"MLP Fold {fold}: "
            f"Accuracy = {accuracy:.4f}, F1 = {f1:.4f}"
        )

    final_val_losses = [h["val_loss"][-1] for h in histories]

    results = {
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores),
        "mean_val_loss": np.mean(final_val_losses),
        "std_val_loss": np.std(final_val_losses),
        "histories": histories,
    }

    print_cross_validation_metrics(results)
    return results



def evaluate_test_mlp(
    train_features, train_labels, test_features, test_labels):
    model, scaler = partial_fit_mlp(train_features, train_labels)
    

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

    # Per-class metrics
    print("\n--- Per-Class Classification Report ---")
    print(classification_report(test_labels, predictions, digits=4))


def evaluate_test_mlp_hpo(
    test_features, test_labels, model, scaler):

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

    # Per-class metrics
    print("\n--- Per-Class Classification Report ---")
    print(classification_report(test_labels, predictions, digits=4))



def plot_mlp_confusion_matrix(
    train_features, train_labels, test_features, test_labels):
    model, scaler = fit_mlp(train_features, train_labels)
    X_test_scaled = scaler.transform(test_features)

    predictions = model.predict(X_test_scaled)
    cm = confusion_matrix(test_labels, predictions, normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format=".2f")
    plt.title("MLP Classifier - Normalized Test Confusion Matrix")
    plt.xlabel("Predicted Speaker Class")
    plt.ylabel("True Speaker Class")
    plt.tight_layout()
    plt.show()


def plot_mlp_loss_curves(histories):
    # Collect losses from all folds
    train_losses = np.array([
        h["train_loss"] for h in histories
    ])
    val_losses = np.array([
        h["val_loss"] for h in histories
    ])

    # Calculate mean and standard deviation
    mean_train = np.mean(train_losses, axis=0)
    mean_val = np.mean(val_losses, axis=0)

    std_train = np.std(train_losses, axis=0)
    std_val = np.std(val_losses, axis=0)

    epochs = np.arange(1, len(mean_train) + 1)

    plt.figure(figsize=(10, 6))

    # Training loss
    plt.plot(epochs, mean_train, label="Training Loss")
    plt.fill_between(
        epochs,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.15,
    )

    # Validation loss
    plt.plot(epochs, mean_val, label="Validation Loss")
    plt.fill_between(
        epochs,
        mean_val - std_val,
        mean_val + std_val,
        alpha=0.15,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MLP Training vs Validation Loss (5-Fold CV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def print_cross_validation_metrics(results):
    print("\n--- Cross-Validation Metrics ---")
    print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std Accuracy:  {results['std_accuracy']:.4f}")
    print(f"Mean Macro F1: {results['mean_f1']:.4f}")
    print(f"Std Macro F1:  {results['std_f1']:.4f}")
    if "mean_val_loss" in results and "std_val_loss" in results:
        print(f"Mean Final Validation Loss: {results['mean_val_loss']:.4f}")
        print(f"Std Final Validation Loss:  {results['std_val_loss']:.4f}")
    print("\n")
    print("--------------------------------------------------------------")