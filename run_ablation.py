import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.data_import import create_labels, load_sequences
from src.preprocessing import create_folds, extract_features


def run_full_grid_search():
    # We load data exactly like in main.py
    data_dir = Path(__file__).resolve().parent / "data"

    print("Loading sequences and labels...")
    train_inputs = load_sequences(data_dir / "ae.train")
    test_inputs = load_sequences(data_dir / "ae.test")
    train_labels, test_labels = create_labels()

    # Extract features
    print("Extracting features...")
    train_features = extract_features(train_inputs)
    test_features = extract_features(test_inputs)

    # Define the grid (36 configurations total 3x3x4 = 36 :)
    layer_options = [(16,), (32,), (64,)]
    activation_options = ["relu", "tanh", "logistic"]
    alpha_options = [0.5, 0.1, 0.01, 0.001]

    results = []

    for layers, act, alpha in itertools.product(
        layer_options, activation_options, alpha_options
    ):
        # the 5-Fold Cross Validation
        cv_accuracies = []
        cv_f1_scores = []

        for train_idx, val_idx in create_folds(train_features, train_labels):
            X_train, X_val = train_features[train_idx], train_features[val_idx]
            y_train, y_val = train_labels[train_idx], train_labels[val_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

            model = MLPClassifier(
                hidden_layer_sizes=layers,
                activation=act,
                alpha=alpha,
                solver="adam",
                max_iter=1000,
                random_state=42,
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            cv_accuracies.append(accuracy_score(y_val, preds))
            cv_f1_scores.append(f1_score(y_val, preds, average="macro"))

        mean_cv_acc = np.mean(cv_accuracies)
        mean_cv_f1 = np.mean(cv_f1_scores)

        # testing the set Evaluation for this config
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_features)
        X_test_scaled = scaler.transform(test_features)

        final_model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation=act,
            alpha=alpha,
            solver="adam",
            max_iter=1000,
            random_state=42,
        )
        final_model.fit(X_train_scaled, train_labels)
        test_preds = final_model.predict(X_test_scaled)

        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro")

        config_str = f"Layers={str(layers):<8} | Act={act:<8} | alpha={alpha}"
        results.append(
            {
                "Config": config_str,
                "CV Acc": mean_cv_acc,
                "CV F1": mean_cv_f1,
                "Test Acc": test_acc,
                "Test F1": test_f1,
            }
        )
        print(
            f"Tested => Layers={layers}, Act={act}, alpha={alpha} | Test Acc: {test_acc:.4f}"
        )

    # clean final sorted summary table
    results = sorted(results, key=lambda x: x["Test Acc"], reverse=True)

    print("\n" + "=" * 95)
    print(
        f"{'Configuration':<38} | {'CV Acc':<10} | {'CV F1':<10} | {'Test Acc':<10} | {'Test F1':<10}"
    )
    print("-" * 95)
    for r in results:
        print(
            f"{r['Config']:<38} | {r['CV Acc']:.4f}     | {r['CV F1']:.4f}     | {r['Test Acc']:.4f}     | {r['Test F1']:.4f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    run_full_grid_search()
