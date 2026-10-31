from itertools import product

from src.evaluate import cross_validate_mlp


def optimize_mlp(features, labels):
    hidden_sizes = [(16,), (32,), (64,)]
    alphas = [0.001,0.01, 0.1, 0.5]
    activations = ["tanh", "relu", "logistic"]
    batch_sizes = [16, 32, 64]

    best_score = -float("inf")
    best_params = None
    best_results = None
    run_count = 0

    for hidden_size, alpha, activation, batch_size in product(
        hidden_sizes, alphas, activations, batch_sizes
    ):
        params = {
            "hidden_layer_sizes": hidden_size,
            "alpha": alpha,
            "activation": activation,
            "batch_size": batch_size,
        }
        run_count += 1
        print(
            f"\nRun {run_count}: "
            f"hidden_layer_sizes={hidden_size}, "
            f"alpha={alpha}, "
            f"activation={activation}, "
            f"batch_size={batch_size}"
        )
        results = cross_validate_mlp(
            features,
            labels,
            epochs=300,
            **params,
        )

        score = results["mean_f1"]

        if score > best_score:
            best_score = score
            best_params = params
            best_results = results

    print("Best hyperparameters:", best_params)
    print(f"Best mean macro F1: {best_score:.4f}")

    return best_params, best_results
