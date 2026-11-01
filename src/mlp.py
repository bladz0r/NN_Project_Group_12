from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def create_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="logistic",
        alpha=0.01,
        solver="adam",
        learning_rate_init=0.001,
        batch_size=16,
        max_iter=1000,
        random_state=42,
    )