import numpy as np
from pathlib import Path
from src.data_import import load_sequences, create_labels
from src.data_exploration import (
    explore_data, plot_recording_lengths, plot_recording_lpcc, explore_lpcc, 
    plot_lpcc_distributions, plot_class_distribution, plot_recording_lengths_by_speaker,
    plot_lpcc_correlations
)
from src.preprocessing import preprocess_data, create_folds

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




# ---------- Data Preprocessing without Cross-Validation ----------
# X_train, X_val, y_train, y_val = preprocess_data(
#     train_inputs,
#     train_labels
# )



# ---------- Data Preprocessing with Cross-Validation ----------
for train_idx, val_idx in create_folds(train_inputs, train_labels):

    X_train = [train_inputs[i] for i in train_idx]
    X_val = [train_inputs[i] for i in val_idx]

    y_train = train_labels[train_idx]
    y_val = train_labels[val_idx]

    # print(len(X_train), len(X_val))


# ----------- Verify the shapes of the training and validation sets -----------
print("Training recordings:", len(X_train))
print("Validation recordings:", len(X_val))

print("Training labels:", np.bincount(y_train))
print("Validation labels:", np.bincount(y_val))