import numpy as np

def load_sequences(filepath):
    data = np.loadtxt(filepath)

    sequences = []
    sequence = []

    for row in data:
        if row[0] == 1:
            if sequence:
                sequences.append(np.array(sequence))
            sequence = []
        else:
            sequence.append(row)

    if sequence:
        sequences.append(np.array(sequence))

    return sequences


def create_labels():
    train_labels = np.repeat(np.arange(9), 30)

    test_lengths = [31, 35, 88, 44, 29, 24, 40, 50, 29]
    test_labels = np.repeat(np.arange(9), test_lengths)

    return train_labels, test_labels