import numpy as np
import matplotlib.pyplot as plt

FRAME_LENGTH = 25.6 # ms
FRAME_SHIFT = 6.4 # ms

def explore_data(inputs, labels):
    lengths = [len(seq) for seq in inputs]

    print("Recordings:", len(inputs))
    print("Features:", inputs[0].shape[1])
    print("Speakers:", len(np.unique(labels)))
    print("Average length (frames):", np.mean(lengths))
    print("Shortest (frames):", min(lengths))
    print("Longest (frames):", max(lengths))



def plot_recording_lengths(inputs):
    lengths = [FRAME_LENGTH + (len(seq) - 1) * FRAME_SHIFT for seq in inputs]

    plt.hist(lengths, bins=30)
    plt.xlabel("Recording length (ms)")
    plt.ylabel("Recordings")
    plt.title("Recording Length Distribution")
    plt.show()


def plot_recording_lpcc(inputs, index=0):
    recording = inputs[index]
    
    time = np.arange(len(recording)) * FRAME_SHIFT

    plt.plot(time, recording)
    plt.xlabel("Time (ms)")
    plt.ylabel("LPCC value")
    plt.title(f"Recording {index + 1} - LPCCs")
    plt.show()


def explore_lpcc(inputs):
    data = np.vstack(inputs)

    for i in range(data.shape[1]):
        feature = data[:, i]
        
        print(f"LPCC {i + 1}")
        print(f"  Mean: {np.mean(feature):.4f}")
        print(f"  Std:  {np.std(feature):.4f}")
        print(f"  Min:  {np.min(feature):.4f}")
        print(f"  Max:  {np.max(feature):.4f}")



def plot_lpcc_distributions(inputs):
    data = np.vstack(inputs)

    plt.boxplot(data)
    plt.xlabel("LPCC index")
    plt.ylabel("LPCC value")
    plt.title("LPCC Distributions")
    plt.show()



def plot_class_distribution(train_labels, test_labels):
    train_counts = np.bincount(train_labels)
    test_counts = np.bincount(test_labels)

    classes = np.arange(1, len(train_counts) + 1)

    plt.figure()
    plt.bar(classes - 0.2, train_counts, width=0.4, label="Train")
    plt.bar(classes + 0.2, test_counts, width=0.4, label="Test")

    plt.xlabel("Speaker")
    plt.ylabel("Number of recordings")
    plt.title("Class Distribution: Train vs Test")
    plt.xticks(classes)
    plt.legend()
    plt.show()



def plot_recording_lengths_by_speaker(inputs, labels):
    lengths = np.array([
        FRAME_LENGTH + (len(seq) - 1) * FRAME_SHIFT
        for seq in inputs
    ])

    speaker_lengths = [
        lengths[labels == i] for i in range(9)
    ]

    plt.boxplot(speaker_lengths)
    plt.xlabel("Speaker")
    plt.ylabel("Recording length (ms)")
    plt.title("Recording Lengths by Speaker")
    plt.xticks(range(1, 10))
    plt.show()



def plot_lpcc_correlations(inputs):
    data = np.vstack(inputs)

    correlations = np.corrcoef(data, rowvar=False)

    plt.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")

    plt.xticks(range(12), range(1, 13))
    plt.yticks(range(12), range(1, 13))

    plt.xlabel("LPCC index")
    plt.ylabel("LPCC index")
    plt.title("LPCC Correlation Matrix")
    plt.show()