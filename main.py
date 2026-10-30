from pathlib import Path

import numpy as np

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
from src.train import (
    evaluate_test_mlp,
    plot_mlp_confusion_matrix,
    train_baseline,
    train_mlp,
)

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


# ---------- Baseline Model Training (Logistic Regression) ----------
train_baseline(train_features, train_labels)


# ---------- MLP Model Training ----------
train_mlp(train_features, train_labels)

# ---------- Final Test Evaluation ----------
evaluate_test_mlp(train_features, train_labels, test_features, test_labels)

# ---------- Confusion Matrix Generation ----------
plot_mlp_confusion_matrix(train_features, train_labels, test_features, test_labels)
