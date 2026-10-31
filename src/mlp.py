from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def create_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        alpha=0.5,
        solver="adam",
        learning_rate_init=0.001,
        batch_size=64,
        max_iter=1000,
        random_state=42,
    )