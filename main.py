from pathlib import Path

import numpy as np

from src.hpo import optimize_mlp
from src.data_exploration import (
    explore_data,
    explore_lpcc,
    plot_class_distribution,
    plot_lpcc_correlations,
    plot_lpcc_distributions,
    plot_recording_lengths,
    plot_recording_lengths_by_speaker,
    plot_recording_lpcc,
)
from src.data_import import create_labels, load_sequences
from src.preprocessing import create_folds, extract_features
from src.evaluate import (cross_validate_baseline, 
    cross_validate_mlp , evaluate_test_mlp,
    plot_mlp_confusion_matrix,
    plot_mlp_loss_curves,evaluate_test_mlp_hpo
)
from src.train import partial_fit_mlp

data_dir = Path(__file__).resolve().parent / "data"

train_inputs = load_sequences(data_dir / "ae.train")
test_inputs = load_sequences(data_dir / "ae.test")

# print("Training recordings:", len(train_inputs))
# print("Testing recordings:", len(test_inputs))
# print("First recording shape:", train_inputs[0].shape)


train_labels, test_labels = create_labels()

# print("First 10 training labels:", train_labels[:50])
# print("First 10 testing labels:", test_labels[:50])


# ---------- Data Exploration ----------
# explore_data(train_inputs, train_labels)
# plot_recording_lengths(train_inputs)
# plot_recording_lpcc(train_inputs, index=2)
# explore_lpcc(train_inputs)
# plot_lpcc_distributions(train_inputs)
# plot_class_distribution(train_labels, test_labels)
# plot_recording_lengths_by_speaker(train_inputs, train_labels)
# plot_lpcc_correlations(train_inputs)


# ---------- Feature Extraction ----------
train_features = extract_features(train_inputs)
test_features = extract_features(test_inputs)

print("Training features:", train_features.shape)
print("Testing features:", test_features.shape)



# ---------- Cross-validate Baseline (Logistic Regression) ----------
cross_validate_baseline(train_features, train_labels)


# ---------- Cross-validate MLP ----------
results = cross_validate_mlp(train_features, train_labels)

# ---------- Plot train vs validation loss curves ----------
# plot_mlp_loss_curves(results["histories"])

# ---------- Final Test Evaluation ----------
evaluate_test_mlp(train_features, train_labels, test_features, test_labels)

# ---------- Confusion Matrix Generation ----------
# plot_mlp_confusion_matrix(train_features, train_labels, test_features, test_labels)



# ------------------ HPO ---------------------------
print("\n--- Hyperparameter Optimization (HPO) ---")

# ------------------ Get best hyperparameters ------------------
# best_params, best_results = optimize_mlp(train_features, train_labels)


# Use this to evaluate the best hyperparameters without having run the HPO again
best_params = {
    "hidden_layer_sizes": (32,),
    "alpha": 0.001,
    "activation": "tanh",
    "batch_size": 32,
}

# ------------------ Train final model with best hyperparameters ------------------
model, scaler = partial_fit_mlp(
    train_features,
    train_labels,
    epochs=300,
    **best_params,
)


# ------------------ Cross-validate final model ------------------
results = cross_validate_mlp(train_features, train_labels, **best_params)

# ------------------ Plot train vs validation loss curves for final model ------------------
plot_mlp_loss_curves(results["histories"])

# ------------------ Final Test Evaluation for final model ------------------
evaluate_test_mlp_hpo(test_features, test_labels, model, scaler)
